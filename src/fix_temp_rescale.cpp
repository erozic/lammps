// clang-format off
/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#include "fix_temp_rescale.h"

#include "atom.h"
#include "comm.h"
#include "compute.h"
#include "error.h"
#include "force.h"
#include "group.h"
#include "input.h"
#include "modify.h"
#include "update.h"
#include "variable.h"

#include <cmath>
#include <cstring>

using namespace LAMMPS_NS;
using namespace FixConst;

enum{NOBIAS,BIAS};
enum{CONSTANT,EQUAL};

/* ---------------------------------------------------------------------- */

FixTempRescale::FixTempRescale(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg),
  tstr(nullptr), id_temp(nullptr), tflag(0)
{
  if (narg < 8) utils::missing_cmd_args(FLERR, "fix temp/rescale", error);

  nevery = utils::inumeric(FLERR,arg[3],false,lmp);
  if (nevery <= 0) error->all(FLERR, "Invalid fix temp/rescale every argument: {}", nevery);

  restart_global = 1;
  scalar_flag = 1;
  global_freq = nevery;
  extscalar = 1;
  ecouple_flag = 1;
  dynamic_group_allow = 1;

  tstr = nullptr;
  if (utils::strmatch(arg[4],"^v_")) {
    tstr = utils::strdup(arg[4]+2);
    tstyle = EQUAL;
  } else {
    t_start = utils::numeric(FLERR,arg[4],false,lmp);
    t_target = t_start;
    tstyle = CONSTANT;
  }

  t_stop = utils::numeric(FLERR,arg[5],false,lmp);
  t_window = utils::numeric(FLERR,arg[6],false,lmp);
  fraction = utils::numeric(FLERR,arg[7],false,lmp);

  if (t_stop < 0) error->all(FLERR, "Invalid fix temp/rescale Tstop argument: {}", t_stop);
  if (t_window < 0) error->all(FLERR, "Invalid fix temp/rescale window argument: {}", t_window);
  if (fraction <= 0) error->all(FLERR, "Invalid fix temp/rescale fraction argument: {}", fraction);

  // create a new compute temp
  // id = fix-ID + temp, compute group = fix group

  id_temp = utils::strdup(std::string(id) + "_temp");
  modify->add_compute(fmt::format("{} {} temp",id_temp,group->names[igroup]));
  tflag = 1;

  energy = 0.0;
}

/* ---------------------------------------------------------------------- */

FixTempRescale::~FixTempRescale()
{
  delete[] tstr;

  // delete temperature if fix created it

  if (tflag) modify->delete_compute(id_temp);
  delete[] id_temp;
}

/* ---------------------------------------------------------------------- */

int FixTempRescale::setmask()
{
  int mask = 0;
  mask |= END_OF_STEP;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixTempRescale::init()
{
  // check variable

  if (tstr) {
    tvar = input->variable->find(tstr);
    if (tvar < 0)
      error->all(FLERR,"Variable name {} for fix {} does not exist", tstr, style);
    if (input->variable->equalstyle(tvar)) tstyle = EQUAL;
    else error->all(FLERR,"Variable {} for fix {} is invalid style", tstr, style);
  }

  temperature = modify->get_compute_by_id(id_temp);
  if (!temperature) {
    error->all(FLERR,"Temperature compute ID {} for fix {} does not exist", id_temp, style);
  } else {
    if (temperature->tempflag == 0)
      error->all(FLERR, "Compute ID {} for fix {} does not compute a temperature", id_temp, style);
    if (temperature->tempbias) which = BIAS;
    else which = NOBIAS;
  }
}

/* ---------------------------------------------------------------------- */

void FixTempRescale::end_of_step()
{
  double t_current = temperature->compute_scalar();

  // there is nothing to do, if there are no degrees of freedom

  if (temperature->dof < 1) return;

  // protect against division by zero

  if (t_current == 0.0)
    error->all(FLERR,"Computed temperature for fix temp/rescale cannot be 0.0");

  double delta = update->ntimestep - update->beginstep;
  if (delta != 0.0) delta /= update->endstep - update->beginstep;

  // set current t_target
  // if variable temp, evaluate variable, wrap with clear/add

  if (tstyle == CONSTANT)
    t_target = t_start + delta * (t_stop-t_start);
  else {
    modify->clearstep_compute();
    t_target = input->variable->compute_equal(tvar);
    if (t_target < 0.0)
      error->one(FLERR, "Fix temp/rescale variable returned negative temperature");
    modify->addstep_compute(update->ntimestep + nevery);
  }

  // rescale velocity of appropriate atoms if outside window
  // for BIAS:
  //   temperature is current, so do not need to re-compute
  //   OK to not test returned v = 0, since factor is multiplied by v

  if (fabs(t_current-t_target) > t_window) {
    t_target = t_current - fraction*(t_current-t_target);
    double factor = sqrt(t_target/t_current);
    double efactor = 0.5 * force->boltz * temperature->dof;

    double **v = atom->v;
    int *mask = atom->mask;
    int nlocal = atom->nlocal;

    energy += (t_current-t_target) * efactor;

    if (which == NOBIAS) {
      for (int i = 0; i < nlocal; i++) {
        if (mask[i] & groupbit) {
          v[i][0] *= factor;
          v[i][1] *= factor;
          v[i][2] *= factor;
        }
      }
    } else {
      for (int i = 0; i < nlocal; i++) {
        if (mask[i] & groupbit) {
          temperature->remove_bias(i,v[i]);
          v[i][0] *= factor;
          v[i][1] *= factor;
          v[i][2] *= factor;
          temperature->restore_bias(i,v[i]);
        }
      }
    }
  }
}

/* ---------------------------------------------------------------------- */

int FixTempRescale::modify_param(int narg, char **arg)
{
  if (strcmp(arg[0],"temp") == 0) {
    if (narg < 2) utils::missing_cmd_args(FLERR, "fix_modify temp", error);
    if (tflag) {
      modify->delete_compute(id_temp);
      tflag = 0;
    }
    delete[] id_temp;
    id_temp = utils::strdup(arg[1]);

    temperature = modify->get_compute_by_id(id_temp);
    if (!temperature)
      error->all(FLERR,"Could not find fix_modify temperature compute {}", id_temp);

    if (temperature->tempflag == 0)
      error->all(FLERR, "Fix_modify temperature compute {} does not compute temperature", id_temp);
    if (temperature->igroup != igroup && comm->me == 0)
      error->warning(FLERR, "Group for fix_modify temp != fix group: {} vs {}",
                     group->names[igroup], group->names[temperature->igroup]);
    return 2;
  }
  return 0;
}

/* ---------------------------------------------------------------------- */

void FixTempRescale::reset_target(double t_new)
{
  t_target = t_start = t_stop = t_new;
}

/* ---------------------------------------------------------------------- */

double FixTempRescale::compute_scalar()
{
  return energy;
}

/* ----------------------------------------------------------------------
   pack entire state of Fix into one write
------------------------------------------------------------------------- */

void FixTempRescale::write_restart(FILE *fp)
{
  int n = 0;
  double list[1];
  list[n++] = energy;

  if (comm->me == 0) {
    int size = n * sizeof(double);
    fwrite(&size,sizeof(int),1,fp);
    fwrite(list,sizeof(double),n,fp);
  }
}

/* ----------------------------------------------------------------------
   use state info from restart file to restart the Fix
------------------------------------------------------------------------- */

void FixTempRescale::restart(char *buf)
{
  int n = 0;
  auto list = (double *) buf;

  energy = list[n++];
}

/* ----------------------------------------------------------------------
   extract thermostat properties
------------------------------------------------------------------------- */

void *FixTempRescale::extract(const char *str, int &dim)
{
  if (strcmp(str,"t_target") == 0) {
    dim = 0;
    return &t_target;
  }
  return nullptr;
}
