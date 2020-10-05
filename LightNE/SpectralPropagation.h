#pragma once

#include "mkl.h"
#include "MKLSVD.h"
#include "ligra.h"
#include "PathEmbed.h"
#include "pbbslib/sequence_ops.h"
#include <boost/math/special_functions/bessel.hpp>
#include <iostream>
#include <cmath>

#define iv(i, s) (boost::math::cyl_bessel_i(i, s))

namespace spectral_propagation {

template <typename FP>
FP* get_embedding_dense(FP* mat, lapack_int n, lapack_int rank, lapack_int dim) {
  assert(dim <= rank);
  // C = U S V^T
  FP* U = new FP[n * rank];
  FP* S = new FP[rank];
  FP* superb = new FP[rank];
  lapack_int lapack_status = mklhelper<FP>::LAPACKE_gesvd(
      LAPACK_ROW_MAJOR,     // int matrix_layout,
      'S',                  // char jobu,
      'N',                  // char jobvt,
      n,                    // lapack_int m,
      rank,                 // lapack_int n,
      mat,                  // float* a,
      rank,                 // lapack_int lda,
      S,                    // float* s,
      U,                    // float* u,
      rank,                 // lapack_int ldu,
      NULL,                 // float* vt,
      n,                    // lapack_int ldvt,
      superb                // float* superb
    );
  assert(lapack_status == 0);
  std::cout << "JacabiSVD for C done." << std::endl;
  FP* emb = path_embed::compute_u_sigma_root<FP>(U, S, n, rank, dim, true, 0.0);
  delete[] U;
  return emb;
}

template <class Graph, typename FP>
FP* chebyshev_expansion(FP* a, Graph& GA, MKL_INT rank, MKL_INT dim, size_t order=10, FP mu=0.2, FP s=0.5) {
  using W = typename Graph::weight_type;
  // I will assume GA has no self loops
  MKL_INT n = static_cast<MKL_INT>(GA.n);
  std::cout << n << std::endl;
  auto offs = pbbs::sequence<MKL_INT>(n+1, [&] (size_t i) { return i==n ? 0 : GA.get_vertex(i).getOutDegree() + 1; });
  size_t m = pbbslib::scan_add_inplace(offs.slice());
  assert(GA.m + GA.n == m);
  std::cout << offs.size() << std::endl;

  // MKL_INT* row_idx = new MKL_INT[m];
  MKL_INT* col_idx = new MKL_INT[m];
  FP* value = new FP[m];
  parallel_for(0, GA.n, [&] (size_t i) {
    size_t k = 0;
    size_t off_i = offs[i];
    FP degree = static_cast<FP>(GA.get_vertex(i).getOutDegree() + 1);
    // put self loop first
    // row_idx[off_i] = i
    col_idx[off_i] = i;
    value[off_i] = 1.0 - mu - 1.0 / degree;
    auto map_f = [&] (const uintE& u, const uintE& v, const W& wgh) {
      assert(u != v);
      ++k;
      // row_idx[off_i + k] = u;
      col_idx[off_i + k] = v;
      value[off_i + k] = -1.0 / degree;
    };
    GA.get_vertex(i).mapOutNgh(i, map_f, false);
  });

  sparse_status_t mkl_status;
  sparse_matrix_t cooM, csrM;
  MKL_INT* rows_start = offs.to_array();
  MKL_INT* rows_end = rows_start + 1;
  mkl_status = mklhelper<FP>::mkl_sparse_create_csr(&csrM,
      SPARSE_INDEX_BASE_ZERO,
      n,
      n,
      rows_start,
      rows_end,
      col_idx,
      value);
  // mkl_status = mkl_sparse_convert_csr(cooM,
  //     SPARSE_OPERATION_NON_TRANSPOSE,
  //     &csrM);
  mkl_status = mkl_sparse_optimize ( csrM );

  FP* Lx0 = new FP[n * rank];
  mklhelper<FP>::cblas_copy(n*rank, a, 1, Lx0, 1);
  struct matrix_descr descrM;
  descrM.type = SPARSE_MATRIX_TYPE_GENERAL;
  FP* tmp = new FP[n * rank];
  std::cout << "going to call mkl_sparse_s_mm" << std::endl;
  // tmp = M * a
  mkl_status = mklhelper<FP>::mkl_sparse_mm(SPARSE_OPERATION_NON_TRANSPOSE,
                      1.,
                      csrM,
                      descrM,
                      SPARSE_LAYOUT_ROW_MAJOR,
                      a,
                      rank,   // number of right hand sides
                      rank,      // ldx
                      0.,
                      tmp,
                      rank);
  assert(mkl_status == SPARSE_STATUS_SUCCESS);
  // TODO Lx1 = 0.5(alpha) * M * tmp + (-1) * a
  FP* Lx1 = new FP[n * rank];
  mklhelper<FP>::cblas_copy(n*rank, a, 1, Lx1, 1);
  mkl_status = mklhelper<FP>::mkl_sparse_mm(SPARSE_OPERATION_NON_TRANSPOSE,
                      0.5,
                      csrM,
                      descrM,
                      SPARSE_LAYOUT_ROW_MAJOR,
                      tmp,
                      rank,   // number of right hand sides
                      rank,      // ldx
                      -1.0,
                      Lx1,
                      rank);


  FP* conv = new FP[n * rank]();
  // conv = iv(0, s) * Lx0  - 2 * iv(1, s) * Lx1
  // first compute iv(0, s) * Lx0
  mklhelper<FP>::cblas_axpy(
      n*rank,     // const MKL_INT n,
      iv(0, s),   // const float a,
      Lx0,        // const float *x,
      1,          // const MKL_INT incx,
      conv,       // float *y,
      1           // const MKL_INT incy);
  );
  // then compute iv(1, s) * Lx1
  mklhelper<FP>::cblas_axpy(
      n*rank,           // const MKL_INT n,
      -2 * iv(1, s),    // const float a,
      Lx1,              // const float *x,
      1,                // const MKL_INT incx,
      conv,             // float *y,
      1                 // const MKL_INT incy);
  );

  FP* Lx2 = new FP[n * rank];
  for (size_t i=2; i<order; ++i) {
    // tmp = M * Lx1
    mkl_status = mklhelper<FP>::mkl_sparse_mm(SPARSE_OPERATION_NON_TRANSPOSE,
                        1.,
                        csrM,
                        descrM,
                        SPARSE_LAYOUT_ROW_MAJOR,
                        Lx1,
                        rank,   // number of right hand sides
                        rank,      // ldx
                        0.,
                        tmp,
                        rank);
    assert(mkl_status == SPARSE_STATUS_SUCCESS);
    // Lx2 = M * tmp - 2 * Lx1 - Lx0
    mklhelper<FP>::cblas_copy(n*rank, Lx0, 1, Lx2, 1);
    // first compute M * tmp - Lx0, store it at Lx2
    mkl_status = mklhelper<FP>::mkl_sparse_mm(SPARSE_OPERATION_NON_TRANSPOSE,
                        1.,
                        csrM,
                        descrM,
                        SPARSE_LAYOUT_ROW_MAJOR,
                        tmp,
                        rank,   // number of right hand sides
                        rank,      // ldx
                        -1.0,
                        Lx2,
                        rank);
    assert(mkl_status == SPARSE_STATUS_SUCCESS);
    // then compute Lx2 = -2 * Lx1 + Lx2
    mklhelper<FP>::cblas_axpy(
        n*rank,           // const MKL_INT n,
        -2.0,             // const float a,
        Lx1,              // const float *x,
        1,                // const MKL_INT incx,
        Lx2,              // float *y,
        1                 // const MKL_INT incy);
    );
    // then comupte conv = (i % 2 == 1 ? -2 : 2) * iv(i, s) * Lx2 + conv
    mklhelper<FP>::cblas_axpy(
        n*rank,                                   // const MKL_INT n,
        ((i&1) ? -2 : 2) * iv(i, s),              // const float a,
        Lx2,                                      // const float *x,
        1,                                        // const MKL_INT incx,
        conv,                                     // float *y,
        1                                         // const MKL_INT incy);
    );
    std::swap(Lx0, Lx1);
    std::swap(Lx1, Lx2);
  }
  mklhelper<FP>::cblas_axpy(
      n*rank,                                   // const MKL_INT n,
      -1.0,                                     // const float a,
      conv,                                     // const float *x,
      1,                                        // const MKL_INT incx,
      a,                                        // float *y,
      1                                         // const MKL_INT incy);
  );

  parallel_for(0, GA.n, [&] (size_t i) {
    // put self loop first
    for (MKL_INT j=0; j<rank; ++j) {
      tmp[i * rank + j] = a[i * rank + j];
    }
    auto map_f = [&] (const uintE& u, const uintE& v, const W& wgh) {
      for (MKL_INT j=0; j<rank; ++j) {
        tmp[u * rank + j] += a[v * rank + j];
      }
    };
    GA.get_vertex(i).mapOutNgh(i, map_f, false);
  });
  delete[] Lx0;
  delete[] Lx1;
  delete[] Lx2;
  delete[] conv;
  FP* emb = get_embedding_dense<FP>(tmp, n, rank, dim);
  delete[] tmp;
  return emb;
}

} // namespace spectral_propagation
