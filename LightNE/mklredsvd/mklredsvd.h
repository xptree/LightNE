#pragma once

#include "mkl.h"
#include "mklhelper.h"

namespace mkl_redsvd {

template <typename FP>
class MKLRedSVD {
public:
  MKLRedSVD(MKL_INT n, MKL_INT nnz, MKL_INT* row_idx, MKL_INT* col_idx, FP* value, bool upper, MKL_INT rank, bool analysis, bool random_project_only=false, bool sparse_project=false, float sparse_project_s=100.0);
  MKLRedSVD(MKL_INT n, MKL_INT* rows_start, MKL_INT* rows_end, MKL_INT* col_indx, FP* value, bool upper, MKL_INT rank, bool analysis, bool random_project_only=false, bool sparse_project=false, float sparse_project_s=100.0);
  ~MKLRedSVD();
  void run();
  FP* S;
  FP* matU;
private:
  sparse_matrix_t csrA;
  MKL_INT n, rank;
  struct matrix_descr descrA;
  bool analysis;
  bool upper;
  bool random_project_only;
  bool sparse_project;
  float sparse_project_s; // Eq.2 of https://web.stanford.edu/~hastie/Papers/Ping/KDD06_rp.pdf
};

} // namespace mkl_redsvd

#include "mklredsvd.cc"
