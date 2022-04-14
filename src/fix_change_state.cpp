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
#include "tokenizer.h"
#include "update.h"

#include <cmath>
#include <cctype>
#include <cfloat>
#include <cstring>

#define MAXLINE 256

using namespace LAMMPS_NS;
using namespace FixConst;

/* ---------------------------------------------------------------------- */

FixChangeState::FixChangeState(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg),
  type_list(nullptr), trans_matrix(nullptr), ntrans(nullptr),
  transition(nullptr), idregion(nullptr),
  sqrt_mass_ratio(nullptr), local_atom_list(nullptr),
  random_global(nullptr), random_local(nullptr), c_pe(nullptr)
{
  if (narg < 13) error->all(FLERR,"Illegal fix change/state command");

  dynamic_group_allow = 1;

  vector_flag = 1;
  size_vector = 2;
  global_freq = 1;
  extvector = 0;
  restart_global = 1;
  time_depend = 1;

  //TODO can it be done without forcing reneighboring?
  // ... post_integrate, post_force or end_step maybe ... ??
  // ... using internal variables to check when to run...
  force_reneighbor = 1;
  next_reneighbor = update->ntimestep + 1;

  // set comm size needed by this Fix
  comm_forward = 1;
  //TODO will change with mols...

  //TODO any other flags and variables from "fix.h" ?

  // required args
  nsteps = utils::inumeric(FLERR, arg[3], false, lmp);
  ncycles = utils::inumeric(FLERR, arg[4], false, lmp);
  seed = utils::inumeric(FLERR, arg[5], false, lmp);
  double temperature = utils::numeric(FLERR, arg[6], false, lmp);

  if (nsteps <= 0)
    error->all(FLERR, "Illegal fix change/state command (N <= 0)");
  if (ncycles <= 0)
    error->all(FLERR, "Illegal fix change/state command (M <= 0)");
  if (seed <= 0)
    error->all(FLERR, "Illegal fix change/state command (seed <= 0)");
  if (temperature <= 0.0)
    error->all(FLERR, "Illegal fix change/state command (T <= 0.0)");

  beta = 1.0/(force->boltz*temperature);

  options(narg-7, &arg[7]);

  if (antisymflag) {
    for (int itype = 0; itype < ntypes; itype++) {
      for (int jtype = itype+1; jtype < ntypes; jtype++) {
        double pen_ij = trans_matrix[itype][jtype];
        double pen_ji = trans_matrix[jtype][itype];
        if (std::isfinite(pen_ij) && std::isinf(pen_ji))
          trans_matrix[jtype][itype] = -pen_ij;
        else if (std::isinf(pen_ij) && std::isfinite(pen_ji))
          trans_matrix[itype][jtype] = -pen_ji;
      }
    }
  }

  memory->create(ntrans, ntypes, "change/state:ntrans");
  transition = (penalty_pair **) memory->smalloc(
      ntypes*sizeof(penalty_pair *), "change/state:transition");
  for (int itype = 0; itype < ntypes; itype++) {
    ntrans[itype] = 0;
    for (int jtype = 0; jtype < ntypes; jtype++) {
      if (std::isfinite(trans_matrix[itype][jtype]))
        ntrans[itype]++;
    }
    if (ntrans[itype] > 0) {
      transition[itype] = (penalty_pair *) memory->smalloc(
          ntrans[itype]*sizeof(penalty_pair), "change/state:transition-part");
    } else {
      error->warning(FLERR, "No transitions defined for atom type {}",
          type_list[itype]);
      continue;
    }
    int jcount = 0;
    for (int jtype = 0; jtype < ntypes; jtype++) {
      if (std::isfinite(trans_matrix[itype][jtype])) {
        transition[itype][jcount].stateindex = jtype;
        transition[itype][jcount].penalty = trans_matrix[itype][jtype];
        jcount++;
      }
    }
  }

  if (comm->me == 0) {
    std::string pen_mat_str = "";
    for (int itype = 0; itype < ntypes; itype++) {
      pen_mat_str += fmt::format(" |{:6.2f}|", trans_matrix[itype][0]);
      for (int jtype = 1; jtype < ntypes; jtype++)
        pen_mat_str += fmt::format("{:6.2f}|", trans_matrix[itype][jtype]);
      pen_mat_str += "\n";
    }
    utils::logmesg(lmp, "Transition matrix from file:\n{}", pen_mat_str);
  }

  // random number generator, same for all procs
  random_global = new RanPark(lmp,seed);

  // random number generator, not the same for all procs
  random_local = new RanPark(lmp,seed);

  nattempts = 0;
  nsuccesses = 0;
  local_atom_nmax = 0;
}

/* ---------------------------------------------------------------------- */

FixChangeState::~FixChangeState()
{
  memory->destroy(type_list);
  memory->destroy(trans_matrix);
  memory->destroy(ntrans);
  for (int itype = 0; itype < ntypes; itype++)
    memory->sfree(transition[itype]);
  memory->sfree(transition);
  memory->destroy(local_atom_list);
  memory->destroy(sqrt_mass_ratio);
  if (regionflag) delete [] idregion;
  delete random_global;
  delete random_local;
}

/* ----------------------------------------------------------------------
   Parse optional parameters at end of input line
------------------------------------------------------------------------- */
void FixChangeState::options(int narg, char **arg)
{
  if (narg < 6) error->all(FLERR, "Illegal fix change/state command");

  antisymflag = 0;
  regionflag = 0;
  ke_flag = 0;
  ntypes = 0;
  iregion = -1;

  int iarg = 0;
  while (iarg < narg) {
    if (strcmp(arg[iarg],"types") == 0) {
      if (iarg+3 > narg) error->all(FLERR, "Illegal fix change/state command");
      iarg++;
      while (iarg+ntypes < narg) {
        if (isalpha(arg[iarg+ntypes][0])) break;
        ntypes++;
      }
      if (ntypes < 2)
        error->all(FLERR, "Illegal fix change/state command: < 2 types");
      if (ntypes > atom->ntypes)
        error->all(FLERR, "Illegal fix change/state command: too many types");
      memory->create(type_list, ntypes, "change/state:type_list");
      // makes sense to create and initialise the trans_matrix here too...
      memory->create(trans_matrix, ntypes, ntypes, "change/state:trans_matrix");
      for (int itype = 0; itype < ntypes; itype++) {
        type_list[itype] = utils::inumeric(FLERR, arg[iarg+itype], false, lmp);
        if (type_list[itype] <= 0 || type_list[itype] > atom->ntypes)
          error->all(FLERR, "Illegal fix change/state command: type out of range");
        for (int jtype = 0; jtype < itype; jtype++) {
          if (type_list[jtype] == type_list[itype])
            error->all(FLERR, "Illegal fix change/state command: repeated type");
          trans_matrix[itype][jtype] = trans_matrix[jtype][itype] = INFINITY;
        }
        trans_matrix[itype][itype] = INFINITY;
      }
      iarg += ntypes;
    } else if (strcmp(arg[iarg],"trans_pens") == 0) {
      if (iarg+2 > narg)
        error->all(FLERR, "Illegal fix change/state command (trans_pens)");
      process_transitions_file(arg[iarg+1], 0);
      MPI_Bcast(*trans_matrix, ntypes*ntypes, MPI_DOUBLE, 0, world);
      iarg += 2;
      // optional additional argument to "trans_pens" keyword
      if (iarg < narg && strcmp(arg[iarg],"antisym") == 0) {
        antisymflag = 1;
        iarg++;
      }
    } else if (strcmp(arg[iarg],"region") == 0) {
      if (iarg+2 > narg)
        error->all(FLERR, "Illegal fix change/state command (region)");
      iregion = domain->find_region(arg[iarg+1]);
      if (iregion == -1)
        error->all(FLERR, "Region ID for fix change/state does not exist");
      idregion = utils::strdup(arg[iarg+1]);
      regionflag = 1;
      iarg += 2;
    } else if (strcmp(arg[iarg],"ke") == 0) {
      if (iarg+2 > narg)
        error->all(FLERR,"Illegal fix change/state command (ke)");
      ke_flag = utils::logical(FLERR, arg[iarg+1], false, lmp);
      iarg += 2;
    } else
      error->all(FLERR, "Illegal fix change/state command: unknown option");
  }
}

/* ----------------------------------------------------------------------
   parse the transition penalties file (only proc with given rank)
------------------------------------------------------------------------- */
void FixChangeState::process_transitions_file(const char *filename, int rank)
{
  if (comm->me != rank)
    return;

  FILE *fp = fopen(filename, "r");
  if (fp == nullptr)
    error->one(FLERR, "Cannot open transition penalties file {}: {}",
        filename, utils::getsyserror());

  char rawline[MAXLINE];
  std::string line;
  int ntransitions = 0;
  // skip 1st line of file
  int linenum = 1;
  line = readline(fp, rawline);

  while (true) {
    line = readline(fp, rawline);
    if (line == "EOF")
      break;
    if (line.empty())
      continue; //skip empty lines and comment lines
    try {
      ValueTokenizer values(line);
      if (values.count() != 3)
        error->one(FLERR, "Invalid format of the transition penalties file");
      int typeindex1 = type_index(values.next_int());
      int typeindex2 = type_index(values.next_int());
      double penalty = values.next_double();
      if (typeindex1 < 0 || typeindex2 < 0)
        error->one(FLERR, "Illegal atom type in transition penalties file");
      trans_matrix[typeindex1][typeindex2] = penalty;
      ntransitions++;
    } catch (TokenizerException &e) {
      error->one(FLERR, "Invalid format of the transition penalties file: {}",
          e.what());
    }
  }

  fclose(fp);
}

/* ----------------------------------------------------------------------
   Read a line of text from the input file and return it trimmed
   (without comments and extra spaces).
   Returns "EOF" as an std::string if it reached the end of file.
------------------------------------------------------------------------- */
std::string FixChangeState::readline(FILE *fp, char *rawline)
{
  if (fgets(rawline, MAXLINE, fp) == nullptr) {
    if (feof(fp))
      return "EOF";
    else
      error->one(FLERR, "Unexpected error reading the transition penalties file");
  }
  return utils::trim(utils::trim_comment(rawline));
}

/* ----------------------------------------------------------------------
   Returns the index of "atom type" in the type_list (or -1 if not there)
------------------------------------------------------------------------- */
int FixChangeState::type_index(int atom_type)
{
  int typeindex = -1;
  for (int itype = 0; itype < ntypes; itype++) {
    if (type_list[itype] == atom_type) {
      typeindex = itype;
      break;
    }
  }
  return typeindex;
}

/* ---------------------------------------------------------------------- */
int FixChangeState::setmask()
{
  return PRE_EXCHANGE; //TODO POST_INTEGRATE, POST_FORCE or END_STEP ?
}

/* ---------------------------------------------------------------------- */
void FixChangeState::init()
{
  c_pe = modify->compute[modify->find_compute("thermo_pe")];

  if (ntypes < 2)
    error->all(FLERR, "Illegal fix change/state command: < 2 types");

  // check all mol templates have same charge
  // (irrelevant for atomic simulations - can't change total charge of a single
  // atom, but "charge redistribution" makes sense for molecules)
  if (atom->q_flag) {
    //TODO
  }

  if (ke_flag) {
    memory->create(sqrt_mass_ratio, ntypes, ntypes, "change/state:sqrt_mass_ratio");
    for (int itype = 0; itype < ntypes; itype++) {
      for (int jtype = 0; jtype < ntypes; jtype++) {
        double imass =  atom->mass[type_list[itype]];
        double jmass =  atom->mass[type_list[jtype]];
        sqrt_mass_ratio[itype][jtype] = sqrt(imass/jmass);
      }
    }
  } else {
    double atom_mass = atom->mass[type_list[0]];
    for (int itype = 1; itype < ntypes; itype++) {
      if (atom->mass[type_list[itype]] != atom_mass) {
        if (comm->me == 0)
          error->warning(FLERR,
              "Not all types have same mass (and 'ke' conservation is off)");
        break;
      }
    }
  }

  // check to see if itype and jtype cutoffs are the same (for reneighboring)
  double **cutsq = force->pair->cutsq;
  unequal_cutoffs = false;
  for (int itype = 0; itype < ntypes; itype++){
    for (int jtype = itype + 1; jtype < ntypes; jtype++){
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
}

/* ----------------------------------------------------------------------
   This is where the magic happens...
------------------------------------------------------------------------- */
void FixChangeState::pre_exchange()
{
  // just return if should not be called on this timestep
  if (next_reneighbor != update->ntimestep)
    return;
  //TODO why use (re)neighboring as a trigger?

  energy_stored = energy_full(true);

  update_atom_list();

  // attempt Ncycle atom swaps
  int nsuccess = 0;
  for (int i = 0; i < ncycles; i++) {
    nsuccess += attempt_atom_type_change();
  }

  nattempts += ncycles;
  nsuccesses += nsuccess;

  next_reneighbor = update->ntimestep + nsteps;
  //TODO why use (re)neighboring as a trigger?
}

/* ----------------------------------------------------------------------
   Update the local list of atoms
------------------------------------------------------------------------- */

void FixChangeState::update_atom_list()
{
  int nlocal = atom->nlocal;
  double **x = atom->x;
  int *mask = atom->mask;

  if (atom->nmax > local_atom_nmax) {
    memory->sfree(local_atom_list);
    local_atom_nmax = atom->nmax;
    local_atom_list = (int*)memory->smalloc(local_atom_nmax*sizeof(int),
        "change/state:local_atom_list");
  }

  nparticles_local = 0;

  for (int i = 0; i < nlocal; i++) {
    if (regionflag && domain->regions[iregion]->match(x[i][0], x[i][1], x[i][2]) != 1)
      continue;
    if (mask[i] & groupbit)
      local_atom_list[nparticles_local++] = i;
  }

  MPI_Allreduce(&nparticles_local, &nparticles, 1, MPI_INT, MPI_SUM, world);
  MPI_Exscan(&nparticles_local, &nparticles_before, 1, MPI_INT, MPI_SUM, world);
}

/* ----------------------------------------------------------------------
   Select a random atom

   Returns the local index of atom or -1 if atom not local.
------------------------------------------------------------------------- */
int FixChangeState::random_particle()
{
  int i = -1;
  int iwhichglobal = static_cast<int> (nparticles*random_global->uniform());
  if ((iwhichglobal >= nparticles_before) &&
      (iwhichglobal < nparticles_before + nparticles_local)) {
    int iwhichlocal = iwhichglobal - nparticles_before;
    i = local_atom_list[iwhichlocal];
  }
  return i;
}

/* ----------------------------------------------------------------------
   Attempt a Monte Carlo change of states...

   NOTE: atom charges are assumed equal and so are not updated
------------------------------------------------------------------------- */
int FixChangeState::attempt_atom_type_change()
{
  if (nparticles == 0) return 0;

  double energy_before = energy_stored;

  int oldtype, newtype;
  int oldtypeindex, newtypeindex, trans_index;
  double penalty;

  int i = random_particle();
  if (i >= 0) {
    newtype = oldtype = atom->type[i];
    oldtypeindex = type_index(oldtype);
    if (oldtypeindex < 0)
      error->all(FLERR, "Undeclared atom type found in the fix group");
    if (ntrans[oldtypeindex] == 0)
      return 0; // no possible transitions for this particle...
    trans_index = static_cast<int>(ntrans[oldtypeindex]*random_local->uniform());
    penalty = transition[oldtypeindex][trans_index].penalty;
    newtypeindex = transition[oldtypeindex][trans_index].stateindex;
    newtype = type_list[newtypeindex];
    atom->type[i] = newtype;
  }
  //if (force->kspace) force->kspace->qsum_qsq();
  // only when charges change (maybe for mol? TODO)

  double energy_after = energy_full(false);

  int success = 0;
  if (i >= 0) {
    double boltzmann_factor = exp(beta*(energy_before - energy_after) - penalty);
    if (random_local->uniform() < boltzmann_factor)
      success = 1;
  }
  int success_all = 0;
  MPI_Allreduce(&success, &success_all, 1, MPI_INT, MPI_MAX, world);

  if (success_all) {
    energy_stored = energy_after;
    if (ke_flag && i >= 0) {
      atom->v[i][0] *= sqrt_mass_ratio[oldtypeindex][newtypeindex];
      atom->v[i][1] *= sqrt_mass_ratio[oldtypeindex][newtypeindex];
      atom->v[i][2] *= sqrt_mass_ratio[oldtypeindex][newtypeindex];
    }
  } else {
    if (i >= 0) {
      atom->type[i] = oldtype;
    }
    //if (force->kspace) force->kspace->qsum_qsq();
    // only when charges change (maybe for mol? TODO)
  }
  return success_all;
}

/* ----------------------------------------------------------------------
   Compute system potential energy

   TODO try to make a local change version... (see GCMC)
------------------------------------------------------------------------- */
double FixChangeState::energy_full(bool initial)
{
  int eflag = 1;
  int vflag = 0;

  // ensure all atoms/ghosts in their place, ready for E calc...
  // (usually done in Verlet before reneighboring)
  // if unequal_cutoffs exchange (and reneighboring?) has to be done...
  // (call to comm->exchange() is a no-op but clears ghost atoms)
  // TODO option to not reneighbor even if unequal cutoffs ?? (conditions...)
  if (unequal_cutoffs || initial) {
    if (domain->triclinic) domain->x2lamda(atom->nlocal);
    domain->pbc();
    comm->exchange();
    comm->borders();
    if (domain->triclinic) domain->lamda2x(atom->nlocal + atom->nghost);
    if (modify->n_pre_neighbor) modify->pre_neighbor();
    neighbor->build(1);
    if (modify->n_post_neighbor) modify->post_neighbor(); //unnecessary?
  } else {
    comm->forward_comm(this);
  }

  //force doesn't need to be cleared because it's accumulation
  //doesn't affect energy calculation

  if (modify->n_pre_force) modify->pre_force(vflag);

  if (force->pair && force->pair->compute_flag)
    force->pair->compute(eflag, vflag);

  if (atom->molecular != Atom::ATOMIC) {
    if (force->bond) force->bond->compute(eflag, vflag);
    if (force->angle) force->angle->compute(eflag, vflag);
    if (force->dihedral) force->dihedral->compute(eflag, vflag);
    if (force->improper) force->improper->compute(eflag, vflag);
  }

  if (force->kspace && force->kspace->compute_flag)
    force->kspace->compute(eflag, vflag);

  if (modify->n_pre_reverse) modify->pre_reverse(eflag, vflag);
  //reverse_comm not necessary because forces irrelevant for energy calculation
  if (modify->n_post_force) modify->post_force(vflag);

  update->eflag_global = update->ntimestep;
  double total_energy = c_pe->compute_scalar();

  return total_energy;
}

/* ---------------------------------------------------------------------- */
//TODO for now only type, when mol a lot of other things possibly (charge etc.)
int FixChangeState::pack_forward_comm(int n, int *list, double *buf, int /*pbc_flag*/, int * /*pbc*/)
{
  int i, j, m;
  int *type = atom->type;

  m = 0;
  for (i = 0; i < n; i++) {
    j = list[i];
    buf[m++] = type[j];
  }

  return m;
}

/* ---------------------------------------------------------------------- */
//TODO for now only type, when mol a lot of other things possibly (charge etc.)
void FixChangeState::unpack_forward_comm(int n, int first, double *buf)
{
  int i, m, last;
  int *type = atom->type;

  m = 0;
  last = first + n;
  for (i = first; i < last; i++)
    type[i] = static_cast<int>(buf[m++]);
}

/* ----------------------------------------------------------------------
  Returns:
   0: total number of attempts in the run
   1: total number of successful attempts in the run
------------------------------------------------------------------------- */
double FixChangeState::compute_vector(int n)
{
  if (n == 0) return (double)nattempts;
  if (n == 1) return (double)nsuccesses;
  return 0.0;
}

/* ----------------------------------------------------------------------
   Memory usage of local atom-based arrays
------------------------------------------------------------------------- */
double FixChangeState::memory_usage()
{
  double bytes = (double)local_atom_nmax * sizeof(int);
  return bytes;
}

/* ----------------------------------------------------------------------
   Pack entire state of Fix into one write
------------------------------------------------------------------------- */
void FixChangeState::write_restart(FILE *fp)
{
  int n = 0;
  double list[6];
  list[n++] = random_global->state();
  list[n++] = random_local->state();
  list[n++] = ubuf(next_reneighbor).d;
  list[n++] = ubuf(nattempts).d;
  list[n++] = ubuf(nsuccesses).d;
  list[n++] = ubuf(update->ntimestep).d;

  if (comm->me == 0) {
    int size = n * sizeof(double);
    fwrite(&size,sizeof(int),1,fp);
    fwrite(list,sizeof(double),n,fp);
  }
}

/* ----------------------------------------------------------------------
   Use state info from restart file to restart the Fix
------------------------------------------------------------------------- */
void FixChangeState::restart(char *buf)
{
  int n = 0;
  double *list = (double *) buf;

  seed = static_cast<int> (list[n++]);
  random_global->reset(seed);

  seed = static_cast<int> (list[n++]);
  random_local->reset(seed);

  next_reneighbor = (bigint) ubuf(list[n++]).i;

  nattempts = ubuf(list[n++]).i;
  nsuccesses = ubuf(list[n++]).i;

  bigint ntimestep_restart = (bigint) ubuf(list[n++]).i;
  if (ntimestep_restart != update->ntimestep)
    error->all(FLERR,"Must not reset timestep when restarting fix change/state");
}
