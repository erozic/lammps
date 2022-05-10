// clang-format off
/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
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
#include "molecule.h"
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
#include <algorithm>

#define MAXLINE 256

using namespace LAMMPS_NS;
using namespace FixConst;

/* ---------------------------------------------------------------------- */

FixChangeState::FixChangeState(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg),
  type_list(nullptr), mol_list(nullptr),
  trans_matrix(nullptr), ntrans(nullptr), transition(nullptr),
  region(nullptr), sqrt_mass_ratio(nullptr), local_atom_list(nullptr),
  mol_atom_tag(nullptr), mol_atom_type(nullptr),
  random_global(nullptr), random_local(nullptr), c_pe(nullptr)
{
  if (narg < 12) error->all(FLERR,"Illegal fix change/state command");

  dynamic_group_allow = 1;

  vector_flag = 1;
  size_vector = 2;
  global_freq = 1;
  extvector = 0;
  restart_global = 1;
  time_depend = 1;

  // needs to force it to be called every nsteps
  //  (because "the magic" is in post_neighbor)
  force_reneighbor = 1;
  next_reneighbor = update->ntimestep + 1;

  //TODO any other flags and variables from "fix.h" ?

  int iarg = 3;

  // process args...

  nsteps = utils::inumeric(FLERR, arg[iarg++], false, lmp);
  if (nsteps <= 0)
    error->all(FLERR, "Illegal fix change/state command (N <= 0)");

  ncycles = utils::inumeric(FLERR, arg[iarg++], false, lmp);
  if (ncycles <= 0)
    error->all(FLERR, "Illegal fix change/state command (M <= 0)");

  seed = utils::inumeric(FLERR, arg[iarg++], false, lmp);
  if (seed <= 0)
    error->all(FLERR, "Illegal fix change/state command (seed <= 0)");

  double temperature = utils::numeric(FLERR, arg[iarg++], false, lmp);
  if (temperature <= 0.0)
    error->all(FLERR, "Illegal fix change/state command (T <= 0.0)");

  beta = 1.0/(force->boltz*temperature);
  nstates = 0;
  antisymflag = 0;

  if (strcmp(arg[iarg], "types") == 0) {
    iarg++;
    if (iarg + 2 > narg) error->all(FLERR, "Illegal fix change/state command");
    state_mode = ATOMIC;
    while (iarg + nstates < narg) {
      if (isalpha(arg[iarg+nstates][0])) break;
      nstates++;
    }
    if (nstates < 2)
      error->all(FLERR, "Illegal fix change/state command: < 2 types");
    memory->create(type_list, nstates, "change/state:type_list");
    for (int istate = 0; istate < nstates; istate++) {
      type_list[istate] = utils::inumeric(FLERR, arg[iarg + istate], false, lmp);
      if (type_list[istate] <= 0 || type_list[istate] > atom->ntypes)
        error->all(FLERR, "Illegal fix change/state command: type out of range");
      for (int jstate = 0; jstate < istate; jstate++) {
        if (type_list[jstate] == type_list[istate])
          error->all(FLERR, "Illegal fix change/state command: repeated atom type");
      }
    }
    iarg += nstates;
  } else if (strcmp(arg[iarg], "mols") == 0) {
    iarg++;
    if (iarg + 2 > narg) error->all(FLERR, "Illegal fix change/state command");
    if (atom->molecular == Atom::ATOMIC)
      error->all(FLERR, "Using 'mols' keyword while system is not molecular");
    state_mode = MOLECULAR;
    while (iarg + nstates < narg) {
      if (atom->find_molecule(arg[iarg+nstates]) == -1) break;
      nstates++;
    }
    if (nstates < 2)
      error->all(FLERR, "Illegal fix change/state command: < 2 mols");
    mol_list = (class Molecule **) memory->smalloc(
        nstates * sizeof(class Molecule *), "change/state:mol_list");
    for (int istate = 0; istate < nstates; istate++) {
      int molindex = atom->find_molecule(arg[iarg + istate]);
      if (molindex == -1)
        error->all(FLERR, "Illegal fix change/state command: mol template undefined");
      mol_list[istate] = atom->molecules[molindex];
      if (mol_list[istate]->nset > 1 && comm->me == 0)
        error->warning(FLERR, "Molecule template {} has multiple molecules",
            mol_list[istate]->id);
      for (int jstate = 0; jstate < istate; jstate++)
        if (mol_list[jstate] == mol_list[istate])
          error->all(FLERR, "Illegal fix change/state command: repeated mol template");
    }
    iarg += nstates;
  } else
    error->all(FLERR, "Illegal fix change/state command: "
        "either 'types' or 'mols' is required as first keyword!");

  // set comm size needed by this Fix
  comm_forward = 1;
  if (state_mode == MOLECULAR) {
    if (atom->q_flag)
      comm_forward += 1;
    //TODO bonds, angles, ...
  }

  if (strcmp(arg[iarg], "trans_pens") == 0) {
    iarg++;
    if (iarg + 1 > narg)
      error->all(FLERR, "Illegal fix change/state command (trans_pens)");
    memory->create(trans_matrix, nstates, nstates, "change/state:trans_matrix");
    for (int istate = 0; istate < nstates; istate++)
      for (int jstate = 0; jstate < nstates; jstate++)
        trans_matrix[istate][jstate] = INFINITY;
    process_transitions_file(arg[iarg], 0);
    iarg++;
    MPI_Bcast(*trans_matrix, nstates*nstates, MPI_DOUBLE, 0, world);
    // optional additional argument to "trans_pens" keyword
    if (iarg < narg && strcmp(arg[iarg], "antisym") == 0) {
      iarg++;
      antisymflag = 1;
      for (int istate = 0; istate < nstates; istate++) {
        for (int jstate = istate+1; jstate < nstates; jstate++) {
          double pen_ij = trans_matrix[istate][jstate];
          double pen_ji = trans_matrix[jstate][istate];
          if (std::isfinite(pen_ij) && std::isinf(pen_ji))
            trans_matrix[jstate][istate] = -pen_ij;
          else if (std::isinf(pen_ij) && std::isfinite(pen_ji))
            trans_matrix[istate][jstate] = -pen_ji;
        }
      }
    }
  } else
    error->all(FLERR, "Illegal fix change/state command: "
        "'trans_pens' option is required (after types/mols)!");

  // print transition matrix to log file
  if (comm->me == 0) {
    std::string pen_mat_out = "  Transition matrix from file:\n";
    for (int istate = 0; istate < nstates; istate++) {
      pen_mat_out += fmt::format("    |{:^6.2f}|", trans_matrix[istate][0]);
      for (int jstate = 1; jstate < nstates; jstate++)
        pen_mat_out += fmt::format("{:^6.2f}|", trans_matrix[istate][jstate]);
      pen_mat_out += "\n";
    }
    utils::logmesg(lmp, pen_mat_out);
  }

  // create the transition lists for each state (from the matrix)
  memory->create(ntrans, nstates, "change/state:ntrans");
  transition = (penalty_pair **) memory->smalloc(
      nstates * sizeof(penalty_pair *), "change/state:transition");
  for (int istate = 0; istate < nstates; istate++) {
    ntrans[istate] = 0;
    for (int jstate = 0; jstate < nstates; jstate++) {
      if (std::isfinite(trans_matrix[istate][jstate]))
        ntrans[istate]++;
    }
    if (ntrans[istate] > 0) {
      transition[istate] = (penalty_pair *) memory->smalloc(
          ntrans[istate] * sizeof(penalty_pair), "change/state:transition-part");
    } else {
      transition[istate] = nullptr;
      if (state_mode == MOLECULAR && comm->me == 0)
        error->warning(FLERR, "No transitions defined for molecule template {}",
            mol_list[istate]->id);
      else if (comm->me == 0)
        error->warning(FLERR, "No transitions defined for atom type {}",
            type_list[istate]);
      continue;
    }
    int jcount = 0;
    for (int jstate = 0; jstate < nstates; jstate++) {
      if (std::isfinite(trans_matrix[istate][jstate])) {
        transition[istate][jcount].stateindex = jstate;
        transition[istate][jcount].penalty = trans_matrix[istate][jstate];
        jcount++;
      }
    }
  }

  // process other, non-positional keyword options...
  options(narg-iarg, &arg[iarg]);

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
  memory->sfree(mol_list);
  memory->destroy(trans_matrix);
  memory->destroy(ntrans);
  for (int istate = 0; istate < nstates; istate++)
    memory->sfree(transition[istate]);
  memory->sfree(transition);
  memory->destroy(sqrt_mass_ratio);
  memory->sfree(local_atom_list);
  memory->destroy(mol_atom_tag);
  memory->destroy(mol_atom_type);
  delete random_global;
  delete random_local;
}

/* ----------------------------------------------------------------------
   Parse optional parameters at end of input line
------------------------------------------------------------------------- */
void FixChangeState::options(int narg, char **arg)
{
  regionflag = 0;
  full_flag = 0;
  ke_flag = 0;

  int iarg = 0;
  while (iarg < narg) {
    if (strcmp(arg[iarg], "region") == 0) {
      iarg++;
      if (iarg + 1 > narg)
        error->all(FLERR, "Illegal fix change/state command (region)");
      region = domain->get_region_by_id(arg[iarg]);
      if (!region)
        error->all(FLERR, "Region ID for fix change/state does not exist");
      regionflag = 1;
      iarg++;
    } else if (strcmp(arg[iarg], "full_energy") == 0) {
      iarg++;
      full_flag = 1;
    } else if (strcmp(arg[iarg], "ke") == 0) {
      iarg++;
      if (iarg + 1 > narg)
        error->all(FLERR,"Illegal fix change/state command (ke)");
      ke_flag = utils::logical(FLERR, arg[iarg], false, lmp);
      iarg++;
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
  int stateindex1, stateindex2;
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
      if (state_mode == MOLECULAR) {
        int molindex1 = atom->find_molecule((char*)values.next_string().c_str());
        int molindex2 = atom->find_molecule((char*)values.next_string().c_str());
        if (molindex1 < 0 || molindex2 < 0)
          error->one(FLERR, "Undefined mol state in transition penalties file");
        stateindex1 = mol_state_index(atom->molecules[molindex1]);
        stateindex2 = mol_state_index(atom->molecules[molindex2]);
      } else {
        stateindex1 = atom_state_index(values.next_int());
        stateindex2 = atom_state_index(values.next_int());
      }
      if (stateindex1 < 0 || stateindex2 < 0)
        error->one(FLERR, "Illegal atom/mol state in transition penalties file");
      double penalty = values.next_double();
      trans_matrix[stateindex1][stateindex2] = penalty;
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
   Returns the index of "atom_type" in the type_list (or -1 if not there)
------------------------------------------------------------------------- */
int FixChangeState::atom_state_index(int atom_type)
{
  int stateindex = -1;
  for (int istate = 0; istate < nstates; istate++) {
    if (type_list[istate] == atom_type) {
      stateindex = istate;
      break;
    }
  }
  return stateindex;
}

/* ----------------------------------------------------------------------
   Returns the index of "mol_template" in the mol_list (or -1 if not there)
------------------------------------------------------------------------- */
int FixChangeState::mol_state_index(Molecule *mol_template)
{
  int stateindex = -1;
  for (int istate = 0; istate < nstates; istate++) {
    if (mol_list[istate] == mol_template) {
      stateindex = istate;
      break;
    }
  }
  return stateindex;
}

/* ---------------------------------------------------------------------- */
int FixChangeState::setmask()
{
  return POST_NEIGHBOR;
  // much better than PRE_EXCHANGE (as in gcmc & atom/swap)
  // other possibility is PRE_FORCE (doesn't force reneighboring)
  //  - problem: comes after "force_clear" in Integrate
}

/* ---------------------------------------------------------------------- */
void FixChangeState::init()
{
  if (nstates < 2)
    error->all(FLERR, "Illegal fix change/state command: < 2 states defined");

  if (!full_flag) {
    if ((force->kspace) || (force->pair == nullptr) ||
        (force->pair->single_enable == 0) || (force->pair->tail_flag)) {
      full_flag = 1;
      if (comm->me == 0)
        error->warning(FLERR, "Automatically switched to full_energy option!");
    }
  }

  if (full_flag)
    c_pe = modify->compute[modify->find_compute("thermo_pe")];
  curr_global_pe = NAN; // to be sure to fail if not set

  // Molecule size check & selected molecule group creation

  if (state_mode == MOLECULAR) {
    mol_natoms = mol_list[0]->natoms;
    for (int istate = 1; istate < nstates; istate++) {
      if (mol_list[istate]->natoms != mol_natoms)
        error->all(FLERR, "All molecules have to have the same number of atoms!");
    }
    if (mol_atom_tag) memory->destroy(mol_atom_tag);
    memory->create(mol_atom_tag, mol_natoms, "change/state:mol_atom_tag");
    if (mol_atom_type) memory->destroy(mol_atom_type);
    memory->create(mol_atom_type, mol_natoms, "change/state:mol_atom_type");

    sel_mol_group_id = fmt::format("FixChangeState:{}:SelMolGroup", id);
    group->assign(sel_mol_group_id + " molecule -1");
    sel_mol_group = group->find(sel_mol_group_id);
    if (sel_mol_group == -1)
      error->all(FLERR, "Could not create {}", sel_mol_group_id);
    sel_mol_group_bit = group->bitmask[sel_mol_group];
    sel_mol_group_invbit = sel_mol_group_bit ^ ~0;
  }

  // Mass check (if all same, if ke option on/off)

  double check_mass = state_mass(0);
  for (int istate = 1; istate < nstates; istate++) {
    if (state_mass(istate) != check_mass) {
      check_mass = -1;
      break;
    }
  }
  if (check_mass > 0) {
    if (ke_flag) {
      utils::logmesg(lmp, "NOTE: Disabling ke conservation (all masses equal)\n");
      ke_flag = 0;
    }
  } else if (check_mass < 0) {
    if (ke_flag) {
      memory->create(sqrt_mass_ratio, nstates, nstates, "change/state:sqrt_mass_ratio");
      for (int istate = 0; istate < nstates; istate++) {
        for (int jstate = 0; jstate < nstates; jstate++) {
          double imass = state_mass(istate);
          double jmass = state_mass(jstate);
          sqrt_mass_ratio[istate][jstate] = sqrt(imass/jmass);
        }
      }
    } else if (comm->me == 0)
      error->warning(FLERR,
          "Not all atoms/mols have same mass (and 'ke' conservation is off)");
  }

  // Charge check (if atomic, if molecular)

  if (atom->q_flag) {
    if (state_mode == ATOMIC && comm->me == 0) {
      utils::logmesg(lmp, "NOTE: State change won't change the charges of atoms, only types.\n");
    } else if (state_mode == MOLECULAR) {
      // All mol templates have to have same total charge (redistribution OK, loss/gain NOT OK)
      mol_charge = 0;
      for (int iatom = 0; iatom < mol_natoms; iatom++)
        mol_charge += mol_list[0]->q[iatom];

      for (int istate = 1; istate < nstates; istate++) {
        double state_charge = 0;
        for (int iatom = 0; iatom < mol_natoms; iatom++)
          state_charge += mol_list[istate]->q[iatom];
        if (state_charge != mol_charge) {
          error->all(FLERR, "All molecule templates have to have same total charge!");
        }
      }
    }
  }

  // Cutoff distance check (because reneighboring not done)

  double **cutsq = force->pair->cutsq;
  double min_cutsq = INFINITY, max_cutsq = 0;
  int itype;
  for (int istate = 0; istate < nstates; istate++) {
    if (state_mode == MOLECULAR) {
      for (int iatom = 0; iatom < mol_natoms; iatom++) {
        itype = mol_list[istate]->type[iatom];
        for (int jtype = 1; jtype <= atom->ntypes; jtype++) {
          double cutoff = cutsq[itype][jtype];
          if (cutoff < min_cutsq) min_cutsq = cutoff;
          if (cutoff > max_cutsq) max_cutsq = cutoff;
        }
      }
    } else {
      itype = type_list[istate];
      for (int jtype = 1; jtype <= atom->ntypes; jtype++) {
        double cutoff = cutsq[itype][jtype];
        if (cutoff < min_cutsq) min_cutsq = cutoff;
        if (cutoff > max_cutsq) max_cutsq = cutoff;
      }
    }
  }
  if (std::sqrt(max_cutsq) > std::sqrt(min_cutsq) + neighbor->skin)
    if (comm->me == 0) error->warning(FLERR,
        "Max pair cutoff is larger than min pair cutoff + skin");
}

/* ----------------------------------------------------------------------
   Returns mass of state indicated by index (atom or mol)
------------------------------------------------------------------------- */
double FixChangeState::state_mass(int stateindex)
{
  if (state_mode == MOLECULAR) {
    if (!mol_list[stateindex]->massflag)
      mol_list[stateindex]->compute_mass();
    return mol_list[stateindex]->masstotal;
  } else {
    return atom->mass[type_list[stateindex]];
  }
}

/* ----------------------------------------------------------------------
   This is where the magic happens...
------------------------------------------------------------------------- */
void FixChangeState::post_neighbor()
{
  // just return if should not be called on this timestep
  if (next_reneighbor != update->ntimestep)
    return;

  /*TODO an update_mol_list() method when MOLECULAR ?
   *  1) a map of mol_ID -> list of tags (local to global!?)
   *    - O(N) + a lot of communication & sorting
   *  2) random_molecule from list of keys of map
   *    - O(1), no comm
   *  3) determine_mol_state is fully local then...
   */
  update_atom_list();

  if (full_flag) // initial PE calculation
    curr_global_pe = total_energy_global();

  // attempt Ncycle atom swaps
  int nsuccess = 0;
  for (int i = 0; i < ncycles; i++) {
    if (full_flag) {
      if (state_mode == MOLECULAR)
        nsuccess += attempt_mol_state_change_global();
      else
        nsuccess += attempt_atom_type_change_global();
    } else {
      if (state_mode == MOLECULAR)
        nsuccess += attempt_mol_state_change_local();
      else
        nsuccess += attempt_atom_type_change_local();
    }
  }
  if (full_flag)
    comm->forward_comm(this);

  nattempts += ncycles;
  nsuccesses += nsuccess;

  next_reneighbor = update->ntimestep + nsteps;
}

/* ----------------------------------------------------------------------
   Update the local list of atoms eligible for "state" changing
------------------------------------------------------------------------- */

void FixChangeState::update_atom_list()
{
  int nlocal = atom->nlocal;
  double **x = atom->x;
  int *mask = atom->mask;

  if (atom->nmax > local_atom_nmax) {
    local_atom_nmax = atom->nmax;
    local_atom_list = (int *) memory->srealloc(local_atom_list,
        local_atom_nmax * sizeof(int), "change/state:local_atom_list");
  }

  nparticles_local = 0;
  nparticles_before = 0;

  for (int i = 0; i < nlocal; i++) {
    if (regionflag && region->match(x[i][0], x[i][1], x[i][2]) != 1)
      continue;
    if (mask[i] & groupbit)
      local_atom_list[nparticles_local++] = i;
  }

  MPI_Allreduce(&nparticles_local, &nparticles, 1, MPI_INT, MPI_SUM, world);
  MPI_Exscan(&nparticles_local, &nparticles_before, 1, MPI_INT, MPI_SUM, world);
}

/* ----------------------------------------------------------------------
   Select a random atom (for "state" changing)

   Returns the local index of atom or -1 if atom not local.
------------------------------------------------------------------------- */
int FixChangeState::random_atom()
{
  int i = -1;
  int iwhichglobal = static_cast<int> (nparticles * random_global->uniform());
  if ((iwhichglobal >= nparticles_before) &&
      (iwhichglobal < nparticles_before + nparticles_local)) {
    int iwhichlocal = iwhichglobal - nparticles_before;
    i = local_atom_list[iwhichlocal];
  }
  return i;
}

/* ----------------------------------------------------------------------
   Select a random molecule (for "state" changing)

   Returns the global molecule ID.
------------------------------------------------------------------------- */
tagint FixChangeState::random_molecule()
{
  int i = random_atom();
  tagint mol_id_local = -1;
  if (i >= 0)
    mol_id_local = atom->molecule[i];

  tagint mol_id;
  MPI_Allreduce(&mol_id_local, &mol_id, 1, MPI_LMP_TAGINT, MPI_MAX, world);

  return mol_id;
}

/* ----------------------------------------------------------------------
   Determine the state of a molecule from the properties of atoms that
   comprise it (have the same, given molecular ID).

   NOTE: Currently checks only atom type consistency, i.e. doesn't distinguish
   between two molecules with the same list of atom types (length and order)!

   Returns the index of mol state in mol_list (or -1).

   Also, populates "mol_atom_tag", "mol_atom_type" and the "sel_mol_group"
------------------------------------------------------------------------- */
int FixChangeState::determine_mol_state(tagint mol_id)
{
  if (mol_id <= 0)
    error->all(FLERR, "Found an atom outside a molecule (mol ID <= 0) in the fix group");

  // get all atoms (tags) that make up this molecule (on all procs)

  int natoms, natoms_local = 0, nlocal = atom->nlocal, *mask = atom->mask;
  for (int i = 0; i < nlocal; i++) {
    if (atom->molecule[i] == mol_id) {
      if (natoms_local < mol_natoms)
        mol_atom_tag[natoms_local] = atom->tag[i];
      natoms_local++;
      mask[i] |= sel_mol_group_bit;
    } else {
      mask[i] &= sel_mol_group_invbit;
    }
  }

  MPI_Allreduce(&natoms_local, &natoms, 1, MPI_INT, MPI_SUM, world);
  if (natoms != mol_natoms)
    error->all(FLERR, "Found a molecule larger than the molecule templates!");

  if (comm->me > 0) {
    MPI_Send(mol_atom_tag, natoms_local, MPI_LMP_TAGINT, 0, 17, world);
  } else {
    int offset = natoms_local;
    for (int rank = 1; rank < comm->nprocs; rank++) {
      MPI_Status status;
      int recv_count;
      MPI_Recv(&mol_atom_tag[offset], mol_natoms - offset, MPI_LMP_TAGINT, rank, 17, world, &status);
      MPI_Get_count(&status, MPI_LMP_TAGINT, &recv_count);
      offset += recv_count;
    }
  }
  MPI_Bcast(mol_atom_tag, mol_natoms, MPI_LMP_TAGINT, 0, world);
  std::sort(mol_atom_tag, mol_atom_tag + mol_natoms);

  // get the molecule atom type "signature" (on all procs)

  for (int iatom = 0; iatom < mol_natoms; iatom++) {
    int i = atom->map(mol_atom_tag[iatom]);
    if (i >= 0)
      mol_atom_type[iatom] = atom->type[i];
    else
      mol_atom_type[iatom] = 0;
  }
  MPI_Allreduce(MPI_IN_PLACE, mol_atom_type, mol_natoms, MPI_INT, MPI_MAX, world);

  int stateindex = -1;
  for (int istate = 0; istate < nstates; istate++) {
    stateindex = istate;
    for (int iatom = 0; iatom < mol_natoms; iatom++) {
      // checks only atom type "signature" of molecule
      if (mol_list[istate]->type[iatom] != mol_atom_type[iatom]) {
        stateindex = -1;
        break;
      }
    }
    if (stateindex >= 0)
      return stateindex;
  }
  return stateindex;
}

/* ----------------------------------------------------------------------
   Changes the state of the current molecule (in "mol_atom_tags") according
   to the passed new state "mol_template".

   NOTE: for now only the types and charges of atoms
   TODO bonds, angles, ...
------------------------------------------------------------------------- */
void FixChangeState::change_mol_state(Molecule *mol_template)
{
  for (int iatom = 0; iatom < mol_natoms; iatom++) {
    int i = atom->map(mol_atom_tag[iatom]);
    if (i >= 0) {
      atom->type[i] = mol_template->type[iatom];
      if (atom->q_flag)
        atom->q[i] = mol_template->q[iatom];
    }
  }
}

/* ----------------------------------------------------------------------
   Attempt a Monte Carlo change of atom state/type,
   with local energy calculation

   NOTE: atom charges are assumed equal and so are not updated
------------------------------------------------------------------------- */
int FixChangeState::attempt_atom_type_change_local()
{
  if (nparticles == 0) return 0;

  int oldstate, newstate;
  int oldstateindex, newstateindex, trans_index;
  double energy_before, energy_after, penalty;

  int success = 0;
  int i = random_atom();
  if (i >= 0) {
    oldstate = atom->type[i];
    oldstateindex = atom_state_index(oldstate);
    if (oldstateindex < 0)
      error->one(FLERR, "Undeclared atom type found in the fix group");
    if (ntrans[oldstateindex] == 0)
      return 0; // no possible transitions for this particle...

    energy_before = atom_energy_local(i);

    trans_index = static_cast<int>(ntrans[oldstateindex]*random_local->uniform());
    penalty = transition[oldstateindex][trans_index].penalty;
    newstateindex = transition[oldstateindex][trans_index].stateindex;
    newstate = type_list[newstateindex];

    atom->type[i] = newstate;
    energy_after = atom_energy_local(i);

    double boltzmann_factor = exp(beta*(energy_before - energy_after) - penalty);
    if (random_local->uniform() < boltzmann_factor) {
      success = 1;
      if (ke_flag) {
        atom->v[i][0] *= sqrt_mass_ratio[oldstateindex][newstateindex];
        atom->v[i][1] *= sqrt_mass_ratio[oldstateindex][newstateindex];
        atom->v[i][2] *= sqrt_mass_ratio[oldstateindex][newstateindex];
      }
    } else {
      atom->type[i] = oldstate; // revert change
    }
  }

  int success_all = 0;
  MPI_Allreduce(&success, &success_all, 1, MPI_INT, MPI_MAX, world);

  if (success_all)
    comm->forward_comm(this); // communicate change in type
    // needs to be done after every change because of energy calculations (ghosts)

  return success_all;
}

/* ----------------------------------------------------------------------
   Attempt a Monte Carlo change of molecule state (mol template),
   with local energy calculation
------------------------------------------------------------------------- */
int FixChangeState::attempt_mol_state_change_local()
{
  //TODO fix rigid, rigid/small ??

  if (nparticles == 0) return 0;

  Molecule *oldstate, *newstate;
  int oldstateindex, newstateindex, trans_index;
  double energy_before, energy_after, penalty;

  int success = 0;
  sel_mol_id = random_molecule();
  oldstateindex = determine_mol_state(sel_mol_id);
  if (oldstateindex < 0)
    error->all(FLERR, "Undeclared mol template found in the fix group");
  if (ntrans[oldstateindex] == 0)
    return 0; // no possible transitions for this molecule
  oldstate = mol_list[oldstateindex];

  energy_before = mol_energy_local();

  trans_index = static_cast<int>(ntrans[oldstateindex]*random_global->uniform());
  penalty = transition[oldstateindex][trans_index].penalty;
  newstateindex = transition[oldstateindex][trans_index].stateindex;
  newstate = mol_list[newstateindex];

  change_mol_state(newstate);
  comm->forward_comm(this);
  energy_after = mol_energy_local();
  /*TODO actual state change (and revert) possibly not necessary...
   *     Maybe only use "atom_energy_local" to see how much energy WOULD
   *     change IF state changed...
   */

  double boltzmann_factor = exp(beta*(energy_before - energy_after) - penalty);
  if (random_global->uniform() < boltzmann_factor) {
    success = 1;
    if (ke_flag) {
      double vcm[3], dv[3];
      vcm[0] = vcm[1] = vcm[2] = 0.0;
      group->vcm(sel_mol_group, mol_list[newstateindex]->masstotal, vcm);
      // additive COM velocity correction (to keep relative motion inside molecule)
      dv[0] = vcm[0] * (sqrt_mass_ratio[oldstateindex][newstateindex] - 1);
      dv[1] = vcm[1] * (sqrt_mass_ratio[oldstateindex][newstateindex] - 1);
      dv[2] = vcm[2] * (sqrt_mass_ratio[oldstateindex][newstateindex] - 1);
      for (int iatom = 0; iatom < mol_natoms; iatom++) {
        int i = atom->map(mol_atom_tag[iatom]);
        if (i >= 0) {
          atom->v[i][0] += dv[0];
          atom->v[i][1] += dv[1];
          atom->v[i][2] += dv[2];
        }
      }
    }
  } else {
    change_mol_state(oldstate);
    comm->forward_comm(this);
  }

  return success;
}

/* ----------------------------------------------------------------------
   Attempt a Monte Carlo change of atom state/type,
   with global energy calculation

   NOTE: atom charges are assumed equal and so are not updated
------------------------------------------------------------------------- */
int FixChangeState::attempt_atom_type_change_global()
{
  if (nparticles == 0) return 0;

  int oldstate, newstate;
  int oldstateindex, newstateindex, trans_index;
  double energy_before, energy_after, penalty;

  energy_before = curr_global_pe;

  int i = random_atom();
  if (i >= 0) {
    oldstate = atom->type[i];
    oldstateindex = atom_state_index(oldstate);
    if (oldstateindex < 0)
      error->one(FLERR, "Undeclared atom type found in the fix group");
    if (ntrans[oldstateindex] == 0)
      return 0; // no possible transitions for this particle...

    trans_index = static_cast<int>(ntrans[oldstateindex]*random_local->uniform());
    penalty = transition[oldstateindex][trans_index].penalty;
    newstateindex = transition[oldstateindex][trans_index].stateindex;
    newstate = type_list[newstateindex];
    atom->type[i] = newstate;
  }

  comm->forward_comm(this); // communicate change in type
  energy_after = total_energy_global();

  int success = 0;
  if (i >= 0) {
    double boltzmann_factor = exp(beta*(energy_before - energy_after) - penalty);
    if (random_local->uniform() < boltzmann_factor) {
      success = 1;
      if (ke_flag) {
        atom->v[i][0] *= sqrt_mass_ratio[oldstateindex][newstateindex];
        atom->v[i][1] *= sqrt_mass_ratio[oldstateindex][newstateindex];
        atom->v[i][2] *= sqrt_mass_ratio[oldstateindex][newstateindex];
      }
    } else {
      atom->type[i] = oldstate; // revert change
    }
  }
  int success_all = 0;
  MPI_Allreduce(&success, &success_all, 1, MPI_INT, MPI_MAX, world);

  if (success_all) {
    curr_global_pe = energy_after; // save new energy on all procs
  } else {
    /*
    comm->forward_comm(this);

    * Logically should be here (to communicate revertion to old type) but
    * redundant because will be done before next global energy calculation
    */
  }

  return success_all;
}

/* ----------------------------------------------------------------------
   Attempt a Monte Carlo change of molecule state (mol template),
   with global energy calculation
------------------------------------------------------------------------- */
int FixChangeState::attempt_mol_state_change_global()
{
  //TODO fix rigid, rigid/small ??

  if (nparticles == 0) return 0;

  Molecule *oldstate, *newstate;
  int oldstateindex, newstateindex, trans_index;
  double energy_before, energy_after, penalty;

  energy_before = curr_global_pe;

  int success = 0;
  sel_mol_id = random_molecule();
  oldstateindex = determine_mol_state(sel_mol_id);
  if (oldstateindex < 0)
    error->all(FLERR, "Undeclared mol template found in the fix group");
  if (ntrans[oldstateindex] == 0)
    return 0; // no possible transitions for this molecule
  oldstate = mol_list[oldstateindex];

  trans_index = static_cast<int>(ntrans[oldstateindex]*random_global->uniform());
  penalty = transition[oldstateindex][trans_index].penalty;
  newstateindex = transition[oldstateindex][trans_index].stateindex;
  newstate = mol_list[newstateindex];

  change_mol_state(newstate);
  comm->forward_comm(this); // communicate change in type
  if (force->kspace) // recalculate charge stuff if necessary
    force->kspace->qsum_qsq();
  energy_after = total_energy_global();

  double boltzmann_factor = exp(beta*(energy_before - energy_after) - penalty);
  if (random_global->uniform() < boltzmann_factor) {
    success = 1;
    curr_global_pe = energy_after;
    if (ke_flag) {
      double vcm[3], dv[3];
      vcm[0] = vcm[1] = vcm[2] = 0.0;
      group->vcm(sel_mol_group, mol_list[newstateindex]->masstotal, vcm);
      // additive COM velocity correction (to keep relative motion inside molecule)
      dv[0] = vcm[0] * (sqrt_mass_ratio[oldstateindex][newstateindex] - 1);
      dv[1] = vcm[1] * (sqrt_mass_ratio[oldstateindex][newstateindex] - 1);
      dv[2] = vcm[2] * (sqrt_mass_ratio[oldstateindex][newstateindex] - 1);
      for (int iatom = 0; iatom < mol_natoms; iatom++) {
        int i = atom->map(mol_atom_tag[iatom]);
        if (i >= 0) {
          atom->v[i][0] += dv[0];
          atom->v[i][1] += dv[1];
          atom->v[i][2] += dv[2];
        }
      }
    }
  } else {
    change_mol_state(oldstate);
    /*
    comm->forward_comm(this);
    if (force->kspace)
      force->kspace->qsum_qsq();

    * Logically should be here (to communicate revertion to old type) but
    * redundant because will be done before next global energy calculation
    */
  }

  return success;
}

/* ----------------------------------------------------------------------
   Compute an atom's interaction energy with (local) atoms
------------------------------------------------------------------------- */

double FixChangeState::atom_energy_local(int i)
{
  double **x = atom->x;
  int *type = atom->type;
  double **cutsq = force->pair->cutsq;
  Pair *pair = force->pair;

  double *xi = x[i];
  int itype = type[i];

  double delx, dely, delz, rsq;

  double force = 0.0;
  double energy = 0.0;

  int nall = atom->nlocal + atom->nghost;

  for (int j = 0; j < nall; j++) {

    if (i == j) continue;

    if (state_mode == MOLECULAR) {
      // exclude intramolecular interaction...
      if (atom->molecule[i] == sel_mol_id)
        continue;
      //TODO some kind of option to include intramolecular energy ??
      // ... if yes then half energy (because of double counting)
    }

    double *xj  = x[j];
    int jtype = type[j];

    delx = xi[0] - xj[0];
    dely = xi[1] - xj[1];
    delz = xi[2] - xj[2];
    rsq = delx*delx + dely*dely + delz*delz;

    if (rsq < cutsq[itype][jtype])
      energy += pair->single(i, j, itype, jtype, rsq, 1.0, 1.0, force);
  }

  return energy;
}

/* ----------------------------------------------------------------------
   Compute a molecule's interaction energy with (local) atoms (across all
   procs that own atoms of the molecule).
------------------------------------------------------------------------- */

double FixChangeState::mol_energy_local()
{
  double energy_local = 0.0;
  for (int iatom = 0; iatom < mol_natoms; iatom++) {
    int i = atom->map(mol_atom_tag[iatom]);
    if (i >= 0)
      energy_local += atom_energy_local(i);
  }

  double energy_tot;
  MPI_Allreduce(&energy_local, &energy_tot, 1, MPI_DOUBLE, MPI_SUM, world);

  return energy_tot;
}

/* ----------------------------------------------------------------------
   Compute full system potential energy (over all atoms and processors)
   NOTE: (very) costly...
------------------------------------------------------------------------- */
double FixChangeState::total_energy_global()
{
  int eflag = 1;
  int vflag = 0;

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
  if (modify->n_post_force_any) modify->post_force(vflag);

  update->eflag_global = update->ntimestep;
  double total_energy = c_pe->compute_scalar();

  return total_energy;
}

/* ---------------------------------------------------------------------- */
int FixChangeState::pack_forward_comm(int n, int *list, double *buf, int /*pbc_flag*/, int * /*pbc*/)
{
  int i, j, m;

  m = 0;
  for (i = 0; i < n; i++) {
    j = list[i];
    buf[m++] = atom->type[j];
    if (state_mode == MOLECULAR) {
      if (atom->q_flag)
        buf[m++] = atom->q[j];
      //TODO bonds, angles, ...
    }
  }

  return m;
}

/* ---------------------------------------------------------------------- */
void FixChangeState::unpack_forward_comm(int n, int first, double *buf)
{
  int i, m, last;

  m = 0;
  last = first + n;
  for (i = first; i < last; i++) {
    atom->type[i] = static_cast<int>(buf[m++]);
    if (state_mode == MOLECULAR) {
      if (atom->q_flag)
        atom->q[i] = buf[m++];
      //TODO bonds, angles, ...
    }
  }
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
