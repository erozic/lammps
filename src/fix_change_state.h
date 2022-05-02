/* -*- c++ -*- ----------------------------------------------------------
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

#ifdef FIX_CLASS
// clang-format off
FixStyle(change/state, FixChangeState);
// clang-format on
#else

#ifndef LMP_FIX_CHANGE_STATE_H
#define LMP_FIX_CHANGE_STATE_H

#include "fix.h"

namespace LAMMPS_NS {

class FixChangeState : public Fix {
 public:
  FixChangeState(class LAMMPS *, int, char **);
  ~FixChangeState() override;
  int setmask() override;
  void init() override;
  void post_neighbor() override;
  int pack_forward_comm(int, int *, double *, int, int *) override;
  void unpack_forward_comm(int, int, double *) override;
  double compute_vector(int) override;
  double memory_usage() override;
  void write_restart(FILE *) override;
  void restart(char *) override;

 private:
  int nsteps, ncycles, seed;

  typedef struct penalty_pair { // convenience structure...
      int stateindex; // index of a "state", i.e. an atom type or mol template
      double penalty;
  } penalty_pair;

  enum StateMode {ATOMIC, MOLECULAR};

  int nstates;
  StateMode state_mode;
  int *type_list;
  Molecule **mol_list; // list of pointers to Molecule objects in atom->molecules
  int mol_natoms;
  double mol_charge;
  double **trans_matrix;
  int *ntrans;
  penalty_pair **transition;

  int regionflag; // 0 = anywhere in box, 1 = specific region
  class Region *region; // swap region pointer

  int antisymflag;  // 1 = antisymmetric transition penalty matrix
  int full_flag; // 1 = full (global) PE calc, 0 = single (local) PE calc
  int ke_flag;  // 1 = conserve ke, 0 = do not conserve ke

  int nparticles;  // # of candidates on all procs
  int nparticles_local;  // # of candidates on this proc
  int nparticles_before; // # of candidates on procs < this proc

  int nattempts;
  int nsuccesses;

  double beta;
  double curr_global_pe;
  double **sqrt_mass_ratio;
  int local_atom_nmax;
  int *local_atom_list;
  tagint sel_mol_id; // tag/ID of the "selected" molecule
  tagint *mol_atom_tag; // tags of atoms in the "selected" molecule (sorted)
  int *mol_atom_type; // types of atoms in the "selected" molecule (for template determination)

  class RanPark *random_global;
  class RanPark *random_local;

  class Compute *c_pe;

  void options(int, char**);
  void process_transitions_file(const char*, int);
  std::string readline(FILE*, char*);
  int atom_state_index(int);
  int mol_state_index(Molecule*);
  double state_mass(int);
  void update_atom_list();
  int random_atom();
  tagint random_molecule();
  int determine_mol_state(tagint);
  void change_mol_state(Molecule*);
  int attempt_atom_type_change_local();
  int attempt_mol_state_change_local();
  int attempt_atom_type_change_global();
  int attempt_mol_state_change_global();
  double atom_energy_local(int);
  double mol_energy_local();
  double total_energy_global();
};

}    // namespace LAMMPS_NS

#endif
#endif

/* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running LAMMPS to see the offending line.

E: Using 'mols' keyword while system is not molecular (templated)

Self-explanatory.

E: Molecule template ID does not exist

Self-explanatory.

E: Must specify at least 2 types in fix change/state

Self-explanatory.

E: Region ID for fix change/state does not exist

Self-explanatory.

E: Invalid atom type in fix change/state command

The atom type specified in the command does not exist.

E: Atoms of all types must have the same charge.

Self-explanatory.

E: At least one atom of each type must be present to define charges.

Self-explanatory.

E: Must not reset timestep when restarting fix change/state

Timestep of restart has to be same as current timestep

E: Cannot open transition penalties file

Self-explanatory.

E: Invalid format of the transition penalties file

Self-explanatory.

E: Unexpected error reading the transition penalties file

Self-explanatory.

E: Illegal atom/mol state in transition penalties file

An atom type not stated in the "types" keyword, or
a mol template not stated in the "mols" keyword is used

E: Undeclared atom type found in the fix group

An atom in the fix group has a type not specified by the "types" keyword

W: Molecule template has multiple molecules

Self-explanatory.

W: Not all atoms/mols have same mass (and 'ke' conservation is off)

Self-explanatory.

W: No transitions defined for atom/mol type X

This atom/mol type/template can't transition to any other "state"

W: Max pair cutoff is larger than min pair cutoff + skin

This might cause bad energy calculations (because of missing neighbors)

TODO others...

*/
