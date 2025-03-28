/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef FIX_CLASS
// clang-format off
FixStyle(ave/correlate/long,FixAveCorrelateLong);
// clang-format on
#else

#ifndef LMP_FIX_AVE_CORRELATE_LONG_H
#define LMP_FIX_AVE_CORRELATE_LONG_H

#include "fix.h"

namespace LAMMPS_NS {

class FixAveCorrelateLong : public Fix {
 public:
  FixAveCorrelateLong(class LAMMPS *, int, char **);
  ~FixAveCorrelateLong() override;
  int setmask() override;
  void init() override;
  void setup(int) override;
  void end_of_step() override;

  void write_restart(FILE *) override;
  void restart(char *) override;
  double memory_usage() override;

  double *t;     // Time steps for result arrays
  double **f;    // Result arrays
  unsigned int npcorr;

 private:
  // NOT OPTIMAL: shift2 and accumulator2 only needed in cross-correlations
  double ***shift, ***shift2;
  double ***correlation;
  double **accumulator, **accumulator2;
  unsigned long int **ncorrelation;
  unsigned int *naccumulator;
  unsigned int *insertindex;

  int numcorrelators;    // Recommended 20
  unsigned int p;        // Points per correlator (recommended 16)
  unsigned int m;        // Num points for average (recommended 2; p mod m = 0)
  unsigned int dmin;     // Min distance between ponts for correlators k>0; dmin=p/m

  int length;    // Length of result arrays
  int kmax;      // Maximum correlator attained during simulation

  struct value_t {
    int which;         // type of data: COMPUTE, FIX, VARIABLE
    int argindex;      // 1-based index if data is vector, else 0
    int iarg;          // argument index in original argument list
    std::string id;    // compute/fix/variable ID
    union {
      class Compute *c;
      class Fix *f;
      int v;
    } val;
  };
  std::vector<value_t> values;

  int nvalues, nfreq;
  bigint nvalid, nvalid_last, last_accumulated_step;
  FILE *fp;

  int type, startstep, overwrite;
  bigint filepos;

  int npair;    // number of correlation pairs to calculate
  double *cvalues;

  void accumulate();
  void evaluate();
  bigint nextvalid();

  void add(const int i, const double w, const int k = 0);
  void add(const int i, const double wA, const double wB, const int k = 0);
};

}    // namespace LAMMPS_NS

#endif
#endif
