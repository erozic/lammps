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

  typedef struct penalty_pair {
      int stateindex; // index of a "to" state
      double penalty;
  } penalty_pair;

  enum StateMode {ATOMIC, MOLECULAR};

  int nstates;
  StateMode state_mode;
  int *type_list;
  class Molecule **mol_list; // list of pointers to Molecule objects in atom->molecules
  int mol_natoms;
  double mol_charge;
  double **trans_matrix;
  int antisymflag;  // 1 = antisymmetric transition penalty matrix
  int *ntrans;
  penalty_pair **transition;

  // class NeighList *neigh_list;

  int regionflag; // 0 = anywhere in box, 1 = specific region
  class Region *region; // swap region pointer
  int full_flag; // 1 = full (global) PE calc, 0 = single (local) PE calc
  int skin_flag; // 1 = auto skin adjustment (maxcut - mincut); default 0
  int ke_flag;  // 1 = conserve ke, 0 = do not conserve ke
  int ngroups; // number of additional groups per state
  std::string ***state_group_ids;
  int *state_group_masks;
  int *state_group_invmasks;


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
  int sel_mol_group;
  int sel_mol_group_bit;
  int sel_mol_group_invbit;
  tagint *mol_atom_tag; // sorted "sel_mol" atom tags
  int *mol_atom_type; // sorted "sel_mol" atom types


  class RanPark *random_global;
  class RanPark *random_local;

  class Compute *c_pe;

  // METHODS

  void options(int, char**);
  std::string readline(FILE*, char*);
  void process_transitions_file(const char*, int);

  int atom_state_index(int);
  int mol_state_index(Molecule*);
  double state_mass(int);

  void update_atom_list();

  int random_atom();
  tagint random_molecule();

  int determine_mol_state(tagint);
  void change_mol_state(int, int);

  int attempt_atom_type_change_local();
  int attempt_mol_state_change_local();
  int attempt_atom_type_change_global();
  int attempt_mol_state_change_global();

  double atom_energy_local(int, bool);
  double mol_energy_local();
  double total_energy_global();
};

}    // namespace LAMMPS_NS

#endif
#endif
