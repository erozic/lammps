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
FixStyle(change/state,FixChangeState);
// clang-format on
#else

#ifndef LMP_FIX_CHANGE_STATE_H
#define LMP_FIX_CHANGE_STATE_H

#include "fix.h"

namespace LAMMPS_NS {

class FixChangeState : public Fix {
 public:
  //TODO see if I need any more of these, or if I don't some of these...
  FixChangeState(class LAMMPS *, int, char **);
  ~FixChangeState() override;
  int setmask() override;
  void init() override;
  void pre_exchange() override;
  int pack_forward_comm(int, int *, double *, int, int *) override;
  void unpack_forward_comm(int, int, double *) override;
  double compute_vector(int) override;
  double memory_usage() override;
  void write_restart(FILE *) override;
  void restart(char *) override;

 private:
  int nevery, ncycles, seed;

  int ntypes;
  int *type_list;
  //int nmols;
  //Molecule *mol_list; ??
  int asymflag;  // 0 = symmetric, 1 = asymmetric transition penalty matrix
  double **trans_pens;

  int regionflag; // 0 = anywhere in box, 1 = specific region
  int iregion;  // swap region
  char *idregion; // swap region ID

  int ke_flag;  // yes = conserve ke, no = do not conserve ke

  int nparticles;  // # of candidates on all procs
  int nparticles_local;  // # of candidates on this proc
  int nparticles_before; // # of candidates on procs < this proc

  int nattempts;
  int nsuccesses;

  bool unequal_cutoffs;

  double beta;
  double *qtype;
  double energy_stored;
  double **sqrt_mass_ratio;
  int local_atom_nmax;
  int *local_atom_list;

  class RanPark *random_equal; //TODO ??
  class RanPark *random_unequal; //TODO ??

  class Compute *c_pe; //TODO ??

  void options(int, char **);
  int attempt_change();
  double energy_full();
  int random_particle();
  void update_atom_list();
};

}    // namespace LAMMPS_NS

#endif
#endif

/* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running LAMMPS to see the offending line.

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

E: Cannot do change/state on atoms in atom_modify first group

This is a restriction due to the way atoms are organized in a list to
enable the atom_modify first command. TODO why??

TODO others...

*/
