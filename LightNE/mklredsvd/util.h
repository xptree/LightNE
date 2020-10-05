#pragma once
#include "mkl.h"
#include "mklhelper.h"

namespace mkl_redsvd {

template <typename T>
struct util {
  // static void standard_normal_vec_BM1(VSLStreamStatePtr stream, MKL_INT n, float *x);

  static void standard_normal_vec_BM2(VSLStreamStatePtr stream, MKL_INT n, T *x);

  static void gram_schmidt(lapack_int m, lapack_int n, T* x);
};

} // namespace mkl_redsvd

#include "util.cc"
