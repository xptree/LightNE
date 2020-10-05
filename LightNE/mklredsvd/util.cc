#pragma once
#include "util.h"
#include <cassert>
#include <iostream>

namespace mkl_redsvd {

// void standard_normal_vec_BM1(VSLStreamStatePtr stream, MKL_INT n, float* x) {
//   int err;
//   err = vsRngGaussian(VSL_RNG_METHOD_GAUSSIAN_BOXMULLER,
//       stream,
//       n,
//       x,
//       0.0,
//       1.0);
//   assert(err == VSL_STATUS_OK);
// }

template <typename T>
void util<T>::standard_normal_vec_BM2(VSLStreamStatePtr stream, MKL_INT n, T* x) {
  int err;
  // err = vsRngGaussian(VSL_RNG_METHOD_GAUSSIAN_BOXMULLER2,
  //     stream,
  //     n,
  //     x,
  //     0.0,
  //     1.0);
  err = mklhelper<T>::vRngGaussian(
        VSL_RNG_METHOD_GAUSSIAN_BOXMULLER2,
        stream,
        n,
        x,
        0.0,
        1.0);
  assert(err == VSL_STATUS_OK);
}


template <typename T>
void util<T>::gram_schmidt(lapack_int m, lapack_int n, T* a) {
  // TODO: https://software.intel.com/content/www/us/en/develop/articles/tall-and-skinny-and-short-and-wide-optimizations-for-qr-and-lq-decompositions.html
  // TODO: should use geqr+gemqr for Tall-and-Skinny (TS) matrices
  // https://github.com/numpy/numpy/blob/183fdb290cd46b1f01303d24ac0c9fc3ff24fe05/numpy/linalg/linalg.py#L617-L826
  std::cout << m << " " << n << std::endl;
  lapack_int mn = m < n ? m : n;
  lapack_int err;
  T* tau = new T[mn]();
  std::cout << "sgeqrf..." << std::endl;
  err = mklhelper<T>::LAPACKE_geqrf(
      LAPACK_ROW_MAJOR,
      m,
      n,
      a,
      n,
      tau);
  if (err) {
    std::cout << "sgeqrf returns " << err << std::endl;
    exit(0);
  }
  std::cout << "sorgqr..." << std::endl;
  err = mklhelper<T>::LAPACKE_orgqr(
      LAPACK_ROW_MAJOR,
      m,
      mn,
      mn,
      a,
      n,
      tau
      );
  if (err) {
    std::cout << "sorgqr returns " << err << std::endl;
    exit(0);
  }
  delete[] tau;
}

}
