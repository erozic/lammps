/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/ Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef COMPUTE_CLASS
// clang-format off
ComputeStyle(sna/grid/local,ComputeSNAGridLocal);
// clang-format on
#else

#ifndef LMP_COMPUTE_SNA_GRID_LOCAL_H
#define LMP_COMPUTE_SNA_GRID_LOCAL_H

#include "compute_grid_local.h"

namespace LAMMPS_NS {

class ComputeSNAGridLocal : public ComputeGridLocal {
 public:
  ComputeSNAGridLocal(class LAMMPS *, int, char **);
  ~ComputeSNAGridLocal() override;
  void init() override;
  void compute_local() override;
  double memory_usage() override;

 protected:
  int ncoeff;
  double **cutsq;
  double rcutfac;
  double *radelem;
  double *wjelem;
  int *map;    // map types to [0,nelements)
  int nelements, chemflag;
  int switchinnerflag;
  double *sinnerelem;
  double *dinnerelem;
  class SNA *snaptr;
  double cutmax;
  int quadraticflag;
  double rfac0, rmin0;
  int twojmax, switchflag, bzeroflag, bnormflag, wselfallflag;
  int chunksize;
  int parallel_thresh;
};

}    // namespace LAMMPS_NS

#endif
#endif
