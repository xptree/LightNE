#pragma once

#include "mkl.h"
#include "mklhelper.h"

namespace mkl_redsvd {

template <typename FP>
class MKLRedSVD {
public:
  MKLRedSVD(MKL_INT n, MKL_INT nnz, MKL_INT* row_idx, MKL_INT* col_idx, FP* value, bool upper, MKL_INT rank, bool analysis);
  MKLRedSVD(MKL_INT n, MKL_INT* rows_start, MKL_INT* rows_end, MKL_INT* col_indx, FP* value, bool upper, MKL_INT rank, bool analysis);
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
};

} // namespace mkl_redsvd

#include "mklredsvd.cc"
