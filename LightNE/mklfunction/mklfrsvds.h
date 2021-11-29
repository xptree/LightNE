#pragma once
#include "mkl.h"
#include "mklhelper.h"
#include "mklutil.h"
#include "stopwatch.h"
#include "sp_sign.hpp"

namespace mkl_frsvds {

template <typename FP>
class MKLfrsvds {
public:
  MKLfrsvds(mkl_util::mat_csr<FP> *A,MKL_INT rank,MKL_INT q,MKL_INT s,bool upper,bool analyze, bool random_project_only, size_t sparse_project, float sparse_project_s);
  ~MKLfrsvds();
  void run();
  mkl_util::mat<FP> *S;
  mkl_util::mat<FP> *matU;
  mkl_util::mat<FP> *matV;
private:
  sparse_matrix_t csrA;
  struct matrix_descr descrA;
  MKL_INT n, rank, q, s;
  bool upper;
  bool analyze;
  bool random_project_only;
  size_t sparse_project;
  float sparse_project_s;
};

} // namespace mkl_frsvds
#include "mklfrsvds.cpp"
