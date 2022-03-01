// clang-format off
/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing author: Eugen Rozic (eugen.rozic@irb.hr)
------------------------------------------------------------------------- */

#include "fix_change_state.h"

#include "angle.h"
#include "atom.h"
#include "bond.h"
#include "comm.h"
#include "compute.h"
#include "dihedral.h"
#include "domain.h"
#include "error.h"
#include "fix.h"
#include "force.h"
#include "group.h"
#include "improper.h"
#include "kspace.h"
#include "memory.h"
#include "modify.h"
#include "neighbor.h"
#include "pair.h"
#include "random_park.h"
#include "region.h"
#include "update.h"

#include <cmath>
#include <cctype>
#include <cfloat>
#include <cstring>

using namespace LAMMPS_NS;
using namespace FixConst;

/* ---------------------------------------------------------------------- */

FixChangeState::FixChangeState(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg),
  type_list(nullptr), trans_pens(nullptr), idregion(nullptr),
  qtype(nullptr), sqrt_mass_ratio(nullptr), local_atom_list(nullptr),
  random_equal(nullptr), random_unequal(nullptr), c_pe(nullptr)
{
  if (narg < 13) error->all(FLERR,"Illegal fix atom/swap command");

  dynamic_group_allow = 1;

  vector_flag = 1;
  size_vector = 2;
  global_freq = 1;
  extvector = 0;
  restart_global = 1;
  time_depend = 1;

  // required args
  nevery = utils::inumeric(FLERR, arg[3], false, lmp);
  ncycles = utils::inumeric(FLERR, arg[4], false, lmp);
  seed = utils::inumeric(FLERR, arg[5], false, lmp);
  double temperature = utils::numeric(FLERR, arg[6], false, lmp);

  if (nevery <= 0) error->all(FLERR, "Illegal fix atom/swap command");
  if (ncycles < 0) error->all(FLERR, "Illegal fix atom/swap command");
  if (seed <= 0) error->all(FLERR, "Illegal fix atom/swap command");
  if (temperature <= 0.0) error->all(FLERR, "Illegal fix atom/swap command");

  beta = 1.0/(force->boltz*temperature);

  options(narg-7, &arg[7]);

  // random number generator, same for all procs
  random_equal = new RanPark(lmp,seed);

  // random number generator, not the same for all procs
  random_unequal = new RanPark(lmp,seed);

  //TODO why this?
  force_reneighbor = 1;
  next_reneighbor = update->ntimestep + 1;

  nattempts = 0;
  nsuccesses = 0;
  local_atom_nmax = 0;

  // set comm size needed by this Fix
  if (atom->q_flag) comm_forward = 2;
  else comm_forward = 1;

}

/* ---------------------------------------------------------------------- */

FixChangeState::~FixChangeState()
{
  memory->destroy(type_list);
  memory->destroy(trans_pens);
  memory->destroy(qtype);
  memory->destroy(sqrt_mass_ratio);
  memory->destroy(local_atom_list);
  if (regionflag) delete [] idregion;
  delete random_equal;
  delete random_unequal;
  delete c_pe;
}

/* ----------------------------------------------------------------------
   parse optional parameters at end of input line
------------------------------------------------------------------------- */
void FixChangeState::options(int narg, char **arg)
{
  if (narg < 6) error->all(FLERR, "Illegal fix change/state command");

  asymflag = 0;
  regionflag = 0;
  ke_flag = 1;
  ntypes = 0;
  iregion = -1;

  int iarg = 0;
  while (iarg < narg) {
    if (strcmp(arg[iarg],"types") == 0) {
      if (iarg+3 > narg) error->all(FLERR, "Illegal fix atom/swap command");
      iarg++;
      while (iarg+ntypes < narg) {
        if (isalpha(arg[iarg+ntypes][0])) break;
        ntypes++;
      }
      if (ntypes >= atom->ntypes)
        error->all(FLERR, "Illegal fix atom/swap command: too many types");
      memory->create(type_list, ntypes, "change/state:type_list");
      for (int itype=0; itype < ntypes; itype++){
        type_list[itype] = utils::inumeric(FLERR, arg[iarg+itype], false, lmp);
        if (type_list[itype] <= 0 || type_list[itype] > atom->ntypes)
          error->all(FLERR, "Illegal fix atom/swap command: type out of range");
      }
      iarg += ntypes
    } else if (strcmp(arg[iarg],"trans_pens") == 0) {
      //TODO ...
      iarg += 3;
    } else if (strcmp(arg[iarg],"region") == 0) {
      if (iarg+2 > narg) error->all(FLERR, "Illegal fix atom/swap command");
      iregion = domain->find_region(arg[iarg+1]);
      if (iregion == -1)
        error->all(FLERR, "Region ID for fix atom/swap does not exist");
      idregion = utils::strdup(arg[iarg+1]);
      regionflag = 1;
      iarg += 2;
    } else if (strcmp(arg[iarg],"ke") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix atom/swap command");
      ke_flag = utils::logical(FLERR,arg[iarg+1],false,lmp);
      iarg += 2;
    } else error->all(FLERR, "Illegal fix atom/swap command: unknown option");
  }
}

/* ---------------------------------------------------------------------- */
//TODO
int FixChangeState::setmask()
{
  int mask = 0;
  mask |= PRE_EXCHANGE;
  return mask;
}

/* ---------------------------------------------------------------------- */
//TODO
void FixChangeState::init()
{
  char *id_pe = (char *) "thermo_pe";
  int ipe = modify->find_compute(id_pe);
  c_pe = modify->compute[ipe];

  int *type = atom->type;

  if (ntypes < 2)
    error->all(FLERR, "Must specify at least 2 types in fix change/state");

  //TODO check everything neccessary is defined...

  //check charge
  if (atom->q_flag) {
    double qmax, qmin;
    int firstall, first;
    memory->create(qtype, ntypes, "change/state:qtype");
    for (int itype = 0; itype < ntypes; itype++) {
      first = 1;
      for (int i = 0; i < atom->nlocal; i++) {
        if (!(atom->mask[i] & groupbit))
          continue;
        if (type[i] == type_list[itype]) {
          if (first) {
            qtype[itype] = atom->q[i];
            first = 0;
          } else if (qtype[itype] != atom->q[i]) {
            error->one(FLERR, "Atoms of all types must have the same charge.");
          }
        }
      }
      MPI_Allreduce(&first, &firstall, 1, MPI_INT, MPI_MIN, world);
      if (firstall)
        error->all(FLERR, "At least one atom of each type must be present to define charges.");
      if (first)
        qtype[itype] = -DBL_MAX;
      MPI_Allreduce(&qtype[itype], &qmax, 1, MPI_DOUBLE, MPI_MAX, world);
      if (first)
        qtype[itype] = DBL_MAX;
      MPI_Allreduce(&qtype[itype], &qmin, 1, MPI_DOUBLE, MPI_MIN, world);
      if (qmax != qmin)
        error->all(FLERR, "Atoms of all types must have the same charge.");
    }
  }

  memory->create(sqrt_mass_ratio, atom->ntypes+1, atom->ntypes+1, "change/state:sqrt_mass_ratio");
  for (int itype = 1; itype <= atom->ntypes; itype++)
    for (int jtype = 1; jtype <= atom->ntypes; jtype++)
      sqrt_mass_ratio[itype][jtype] = sqrt(atom->mass[itype]/atom->mass[jtype]);

  // check to see if itype and jtype cutoffs are the same
  // if not, reneighboring will be needed between swaps
  double **cutsq = force->pair->cutsq;
  unequal_cutoffs = false;
  for (int itype = 0; itype < ntypes; itype++){
    for (int jtype = 0; jtype < ntypes; jtype++){
      for (int ktype = 1; ktype <= atom->ntypes; ktype++){
        if (cutsq[type_list[itype]][ktype] != cutsq[type_list[jtype]][ktype]){
          unequal_cutoffs = true;
          break;
        }
      }
      if (unequal_cutoffs) break;
    }
    if (unequal_cutoffs) break;
  }

  // check that no swappable atoms are in atom->firstgroup
  // swapping such an atom might not leave firstgroup atoms first
  //TODO why??
  if (atom->firstgroup >= 0) {
    int *mask = atom->mask;
    int firstgroupbit = group->bitmask[atom->firstgroup];

    int flag = 0;
    for (int i = 0; i < atom->nlocal; i++)
      if ((mask[i] == groupbit) && (mask[i] && firstgroupbit))
        flag = 1;

    int flagall;
    MPI_Allreduce(&flag, &flagall, 1, MPI_INT, MPI_SUM, world);
    if (flagall)
      error->all(FLERR,"Cannot do change/state on atoms in atom_modify first group");
  }
}

/* ----------------------------------------------------------------------
   TODO this is where the magic happens
------------------------------------------------------------------------- */
void FixChangeState::pre_exchange()
{
  // just return if should not be called on this timestep

  if (next_reneighbor != update->ntimestep) return;

  // insure current system is ready to compute energy

  if (domain->triclinic) domain->x2lamda(atom->nlocal);
  domain->pbc();
  comm->exchange();
  comm->borders();
  if (domain->triclinic) domain->lamda2x(atom->nlocal+atom->nghost);
  if (modify->n_pre_neighbor) modify->pre_neighbor();
  neighbor->build(1);

  energy_stored = energy_full();

  // attempt Ncycle atom swaps
  int nsuccess = 0;
  update_atom_list();
  for (int i = 0; i < ncycles; i++)
    nsuccess += attempt_change();

  nattempts += ncycles;
  nsuccesses += nsuccess;

  next_reneighbor = update->ntimestep + nevery;
}

/* ----------------------------------------------------------------------
 * TODO attempt Monte Carlo change of states
   compare before/after energy and accept/reject the change
   NOTE: atom charges are assumed equal and so are not updated
------------------------------------------------------------------------- */
int FixChangeState::attempt_change()
{
  if (nparticles == 0) return 0;

  double energy_before = energy_stored;

  int oldtype, newtype;
  int oldtypeindex, newtypeindex;
  int i = random_particle();
  if (i >= 0) {
    newtype = oldtype = atom->type[i];
    for (int itype = 0; i < ntypes; i++){
      oldtypeindex = itype;
      if (type_list[itype] == oldtype)
        break;
    }
    while (newtype == oldtype) {
      newtypeindex = static_cast<int>(ntypes*random_unequal->uniform());
      newtype = type_list[newtypeindex];
    }
    atom->type[i] = newtype;
    //TODO anything else to change?
  }

  // if unequal_cutoffs, call comm->borders() and rebuild neighbor list
  // else communicate ghost atoms
  // call to comm->exchange() is a no-op but clears ghost atoms
  if (unequal_cutoffs) {
    if (domain->triclinic) domain->x2lamda(atom->nlocal);
    comm->exchange();
    comm->borders();
    if (domain->triclinic) domain->lamda2x(atom->nlocal+atom->nghost);
    if (modify->n_pre_neighbor) modify->pre_neighbor();
    neighbor->build(1);
  } else {
    comm->forward_comm(this);
  }

  if (force->kspace) force->kspace->qsum_qsq(); //TODO ?
  double energy_after = energy_full();

  int success = 0;
  if (i >= 0) {
    double boltzmann_factor = exp(beta*(energy_before - energy_after) -
        trans_pens[oldtypeindex][newtypeindex]);
    if (random_unequal->uniform() < boltzmann_factor)
      success = 1;
  }
  int success_all = 0;
  MPI_Allreduce(&success, &success_all, 1, MPI_INT, MPI_MAX, world);

  if (success_all) {
    update_atom_list();
    energy_stored = energy_after;
    if (ke_flag) {
      if (i >= 0) {
        atom->v[i][0] *= sqrt_mass_ratio[oldtype][newtype];
        atom->v[i][1] *= sqrt_mass_ratio[oldtype][newtype];
        atom->v[i][2] *= sqrt_mass_ratio[oldtype][newtype];
      }
    }
    return 1;
  } else {
    if (i >= 0) atom->type[i] = oldtype;
    if (force->kspace) force->kspace->qsum_qsq(); //TODO why?
    return 0;
  }
}

/* ----------------------------------------------------------------------
   TODO compute system potential energy
   TODO check if optimal...
------------------------------------------------------------------------- */
double FixChangeState::energy_full()
{
  int eflag = 1;
  int vflag = 0;

  if (modify->n_pre_force) modify->pre_force(vflag);

  if (force->pair) force->pair->compute(eflag,vflag);

  if (atom->molecular != Atom::ATOMIC) {
    if (force->bond) force->bond->compute(eflag,vflag);
    if (force->angle) force->angle->compute(eflag,vflag);
    if (force->dihedral) force->dihedral->compute(eflag,vflag);
    if (force->improper) force->improper->compute(eflag,vflag);
  }

  if (force->kspace) force->kspace->compute(eflag,vflag);

  if (modify->n_post_force) modify->post_force(vflag);

  update->eflag_global = update->ntimestep;
  double total_energy = c_pe->compute_scalar();

  return total_energy;
}

/* ----------------------------------------------------------------------
   TODO Return a random atom ID (-1 if not local)
------------------------------------------------------------------------- */
int FixChangeState::random_particle()
{
  int i = -1;
  int iwhichglobal = static_cast<int> (nparticles*random_equal->uniform());
  if ((iwhichglobal >= nparticles_before) &&
      (iwhichglobal < nparticles_before + nparticles_local)) {
    int iwhichlocal = iwhichglobal - nparticles_before;
    i = local_atom_list[iwhichlocal];
  }
  return i;
}

/* ----------------------------------------------------------------------
   TODO update the list of gas atoms
------------------------------------------------------------------------- */

void FixChangeState::update_atom_list()
{
  int nlocal = atom->nlocal;
  double **x = atom->x;

  if (atom->nmax > local_atom_nmax) {
    memory->sfree(local_atom_list);
    local_atom_nmax = atom->nmax;
    local_atom_list = (int*)memory->smalloc(local_atom_nmax*sizeof(int),
        "change/state:local_atom_list");
  }

  nparticles_local = 0;

  if (regionflag) {
    for (int i = 0; i < nlocal; i++) {
      if (domain->regions[iregion]->match(x[i][0], x[i][1], x[i][2]) != 1)
        continue;
      if (!(atom->mask[i] & groupbit))
        continue;
      int atomtype = atom->type[i];
      for (int itype = 0; itype < ntypes; itype++) {
        if (atomtype == type_list[itype]) {
          local_atom_list[nparticles_local] = i;
          nparticles_local++;
          break;
        }
      }
    }
  } else {
    for (int i = 0; i < nlocal; i++) {
      if (!(atom->mask[i] & groupbit))
        continue;
      int atomtype = atom->type[i];
      for (int itype = 0; itype < ntypes; itype++) {
        if (atomtype == type_list[itype]) {
          local_atom_list[nparticles_local] = i;
          nparticles_local++;
          break;
        }
      }
    }
  }

  MPI_Allreduce(&nparticles_local, &nparticles, 1, MPI_INT, MPI_SUM, world);
  MPI_Scan(&nnparticles_local, &nparticles_before, 1, MPI_INT, MPI_SUM, world);
  nparticles_before -= nparticles_local;
}

/* ---------------------------------------------------------------------- */
//TODO
int FixChangeState::pack_forward_comm(int n, int *list, double *buf, int /*pbc_flag*/, int * /*pbc*/)
{
  int i,j,m;

  int *type = atom->type;
  double *q = atom->q;

  m = 0;

  if (atom->q_flag) {
    for (i = 0; i < n; i++) {
      j = list[i];
      buf[m++] = type[j];
      buf[m++] = q[j];
    }
  } else {
    for (i = 0; i < n; i++) {
      j = list[i];
      buf[m++] = type[j];
    }
  }

  return m;
}

/* ---------------------------------------------------------------------- */
//TODO
void FixChangeState::unpack_forward_comm(int n, int first, double *buf)
{
  int i,m,last;

  int *type = atom->type;
  double *q = atom->q;

  m = 0;
  last = first + n;

  if (atom->q_flag) {
    for (i = first; i < last; i++) {
      type[i] = static_cast<int> (buf[m++]);
      q[i] = buf[m++];
    }
  } else {
    for (i = first; i < last; i++)
      type[i] = static_cast<int> (buf[m++]);
  }
}

/* ----------------------------------------------------------------------
  TODO return acceptance ratio
------------------------------------------------------------------------- */
double FixChangeState::compute_vector(int n)
{
  if (n == 0) return (double)nattempts;
  if (n == 1) return (double)nsuccesses;
  return 0.0;
}

/* ----------------------------------------------------------------------
   TODO memory usage of local atom-based arrays
------------------------------------------------------------------------- */
double FixChangeState::memory_usage()
{
  double bytes = (double)local_atom_nmax * sizeof(int);
  return bytes;
}

/* ----------------------------------------------------------------------
   TODO pack entire state of Fix into one write
------------------------------------------------------------------------- */
void FixChangeState::write_restart(FILE *fp)
{
  int n = 0;
  double list[6];
  list[n++] = random_equal->state();
  list[n++] = random_unequal->state();
  list[n++] = ubuf(next_reneighbor).d;
  list[n++] = nattempts;
  list[n++] = nsuccesses;
  list[n++] = ubuf(update->ntimestep).d;

  if (comm->me == 0) {
    int size = n * sizeof(double);
    fwrite(&size,sizeof(int),1,fp);
    fwrite(list,sizeof(double),n,fp);
  }
}

/* ----------------------------------------------------------------------
   TODO use state info from restart file to restart the Fix
------------------------------------------------------------------------- */
void FixChangeState::restart(char *buf)
{
  int n = 0;
  double *list = (double *) buf;

  seed = static_cast<int> (list[n++]);
  random_equal->reset(seed);

  seed = static_cast<int> (list[n++]);
  random_unequal->reset(seed);

  next_reneighbor = (bigint) ubuf(list[n++]).i;

  nattempts = static_cast<int>(list[n++]);
  nsuccesses = static_cast<int>(list[n++]);

  bigint ntimestep_restart = (bigint) ubuf(list[n++]).i;
  if (ntimestep_restart != update->ntimestep)
    error->all(FLERR,"Must not reset timestep when restarting fix atom/swap");
}
