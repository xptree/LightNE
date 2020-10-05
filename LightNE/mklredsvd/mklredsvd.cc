#include "mklredsvd.h"
#include "util.h"
#include <chrono>  // for high_resolution_clock
#include <iostream>
#include <cassert>

using namespace mkl_redsvd;

template <typename FP>
MKLRedSVD<FP>::MKLRedSVD(MKL_INT n, MKL_INT nnz, MKL_INT* row_idx, MKL_INT* col_idx, FP* value, bool upper, MKL_INT rank, bool analysis) {
  std::cout << "init svd from coo" << std::endl;
  this->n = n;
  this->rank = rank;
  this->analysis = analysis;
  this->upper = upper;
  S = NULL;
  matU = NULL;
  sparse_status_t status;
  sparse_matrix_t cooA;
  std::cout << "calling mkl_sparse_s_create_coo" << std::endl;
  auto start = std::chrono::high_resolution_clock::now();
  status = mklhelper<FP>::mkl_sparse_create_coo(&cooA,
      SPARSE_INDEX_BASE_ZERO,
      n,
      n,
      nnz,
      row_idx,
      col_idx,
      value);
  auto finish = std::chrono::high_resolution_clock::now();
  std::chrono::duration<float> elapsed = finish - start;
  std::cout << "Elapsed time: " << elapsed.count() << " s" << std::endl;
  std::cout << "create sparse coo matrix done, status=" << status << " n= " << n << " nnz= " << nnz << std::endl;

  start = std::chrono::high_resolution_clock::now();
  status = mkl_sparse_convert_csr(cooA,
      SPARSE_OPERATION_NON_TRANSPOSE,
      &csrA);
  finish = std::chrono::high_resolution_clock::now();
  elapsed = finish - start;
  std::cout << "Elapsed time: " << elapsed.count() << " s" << std::endl;
  std::cout << "convert to sparse csr matrix done, status=" << status << std::endl;
  status = mkl_sparse_destroy(cooA);
  std::cout << "destroy sparse coo matrix done, status=" << status << std::endl;
}

template <typename FP>
MKLRedSVD<FP>::MKLRedSVD(MKL_INT n, MKL_INT* rows_start, MKL_INT* rows_end, MKL_INT* col_idx, FP* value, bool upper, MKL_INT rank, bool analysis) {
  std::cout << "init svd from csr" << std::endl;
  this->n = n;
  this->rank = rank;
  this->analysis = analysis;
  this->upper = upper;
  S = NULL;
  matU = NULL;
  sparse_status_t status;
  std::cout << "calling mkl_sparse_s_create_csr" << std::endl;
  auto start = std::chrono::high_resolution_clock::now();
  status = mklhelper<FP>::mkl_sparse_create_csr(&csrA,
      SPARSE_INDEX_BASE_ZERO,
      n,
      n,
      rows_start,
      rows_end,
      col_idx,
      value);
  auto finish = std::chrono::high_resolution_clock::now();
  std::chrono::duration<float> elapsed = finish - start;
  std::cout << "Elapsed time: " << elapsed.count() << " s" << std::endl;
  std::cout << "create sparse csr matrix done, status=" << status << " n= " << n << std::endl;
}

template <typename FP>
MKLRedSVD<FP>::~MKLRedSVD() {
  if (S != NULL) {
    delete[] S;
  }
  if (matU != NULL) {
    delete[] matU;
  }
}

template <typename FP>
void MKLRedSVD<FP>::run() {
  if (upper) {
    descrA.type = SPARSE_MATRIX_TYPE_SYMMETRIC;
    descrA.mode = SPARSE_FILL_MODE_UPPER;
    descrA.diag = SPARSE_DIAG_NON_UNIT;
  } else {
    descrA.type = SPARSE_MATRIX_TYPE_GENERAL;
  }

  if (analysis) {
    auto start = std::chrono::high_resolution_clock::now();
    sparse_status_t status = mkl_sparse_set_mm_hint(
          csrA,                               // const sparse_matrix_t A,
          SPARSE_OPERATION_NON_TRANSPOSE,     // const sparse_operation_t operation,
          descrA,                             // const struct matrix_descr descr,
          SPARSE_LAYOUT_ROW_MAJOR,            // const sparse_layout_t layout,
          rank,                               // const MKL_INT dense_matrix_size,
          2                                   // const MKL_INT expected_calls
        );
    std::cout << "set sparse mm hint done, status=" << status << std::endl;
    // status = mkl_sparse_set_memory_hint(csrA, SPARSE_MEMORY_AGGRESSIVE);
    status = mkl_sparse_set_memory_hint(csrA, SPARSE_MEMORY_NONE);
    std::cout << "set sparse memory hint done, status=" << status << std::endl;
    status = mkl_sparse_optimize ( csrA );
    std::cout << "optimize sparse matrix done, status=" << status << std::endl;
    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> elapsed = finish - start;
    std::cout << "Elapsed time (Inspector-executor Sparse BLAS Analysis): " << elapsed.count() << " s" << std::endl;
  }

  VSLStreamStatePtr stream;
  sparse_status_t mkl_status;
  lapack_int lapack_status;
  vslNewStream( &stream, VSL_BRNG_SFMT19937, 777 );
  std::cout << n << std::endl;
  std::cout << rank << std::endl;

  // compute sample matrix O
  FP* O = new FP[n * rank];
  util<FP>::standard_normal_vec_BM2(stream, n * rank, O);
  std::cout << "sampling gaussian random matrix O for A^T done." << std::endl;

  // compute Y = A.T * O = A * O
  FP * Y = new FP[n * rank];

  auto spmm_start = std::chrono::high_resolution_clock::now();
  std::cout << "going to call mkl_sparse_s_mm" << std::endl;
  mkl_status = mklhelper<FP>::mkl_sparse_mm(SPARSE_OPERATION_NON_TRANSPOSE,
                      1.,
                      csrA,
                      descrA,
                      SPARSE_LAYOUT_ROW_MAJOR,
                      O,
                      rank,   // number of right hand sides
                      rank,      // ldx
                      0.,
                      Y,
                      rank);
  assert(mkl_status == SPARSE_STATUS_SUCCESS);
  std::cout << "compute sample matrix of Y = A^T * O  = A * O done (because A^T = A)" << std::endl;
  auto spmm_finish = std::chrono::high_resolution_clock::now();
  std::chrono::duration<float> spmm_elapsed = spmm_finish - spmm_start;
  std::cout << "Elapsed time (spmm): " << spmm_elapsed.count() << " s" << std::endl;
  delete[] O;
  // orthonormalize Y
  std::cout << "going to orthonormalize Y" << std::endl;
  auto gs_start = std::chrono::high_resolution_clock::now();
  util<FP>::gram_schmidt(n, rank, Y);
  std::cout << "orthonormalize Y done." << std::endl;
  auto gs_finish = std::chrono::high_resolution_clock::now();
  std::chrono::duration<float> gs_elapsed = gs_finish - gs_start;
  std::cout << "Elapsed time (gram schmidt): " << gs_elapsed.count() << " s" << std::endl;

  // compute B = A * Y;
  FP* B = new FP[n * rank];
  spmm_start = std::chrono::high_resolution_clock::now();
  mkl_status = mklhelper<FP>::mkl_sparse_mm(SPARSE_OPERATION_NON_TRANSPOSE,
                      1.,
                      csrA,
                      descrA,
                      SPARSE_LAYOUT_ROW_MAJOR,
                      Y,
                      rank,   // number of right hand sides
                      rank,   // ldx
                      0.,
                      B,
                      rank);
  assert(mkl_status == SPARSE_STATUS_SUCCESS);
  std::cout << "B = A * Y done, Range(B) = Range(A^T)." << std::endl;
  spmm_finish = std::chrono::high_resolution_clock::now();
  spmm_elapsed = spmm_finish - spmm_start;
  std::cout << "Elapsed time (spmm): " << spmm_elapsed.count() << " s" << std::endl;
  mkl_sparse_destroy(csrA);

  // sample gaussian random matrix P
  FP* P = new FP[rank * rank];
  util<FP>::standard_normal_vec_BM2(stream, rank * rank, P);
  std::cout << "sample another gaussian random matrix P done." << std::endl;

  FP* Z = new FP[n * rank]();
  // compute Z = B * P
  auto gemm_start = std::chrono::high_resolution_clock::now();
  mklhelper<FP>::cblas_gemm(
      CblasRowMajor, // const CBLAS_LAYOUT Layout,
      CblasNoTrans,  // const CBLAS_TRANSPOSE transa,
      CblasNoTrans,  // const CBLAS_TRANSPOSE transb,
      n,             // const MKL_INT m,
      rank,          // const MKL_INT n,
      rank,          // const MKL_INT k,
      1.0,           // const float alpha,
      B,             // const float *a,
      rank,          // const MKL_INT lda,
      P,             // const float *b,
      rank,          // const MKL_INT ldb,
      0.0,           // const float beta,
      Z,             // float *c,
      rank           // const MKL_INT ldc
  );
  std::cout << "compute sample matrix of Z = B * P done." << std::endl;
  auto gemm_finish = std::chrono::high_resolution_clock::now();
  std::chrono::duration<float> gemm_elapsed = gemm_finish - gemm_start;
  std::cout << "Elapsed time (gemm): " << gemm_elapsed.count() << " s" << std::endl;
  delete[] P;

  // orthonormalize Z
  gs_start = std::chrono::high_resolution_clock::now();
  util<FP>::gram_schmidt(n, rank, Z);
  std::cout << "orthonormalize Z done." << std::endl;
  gs_finish = std::chrono::high_resolution_clock::now();
  gs_elapsed = gs_finish - gs_start;
  std::cout << "Elapsed time (gram schmidt): " << gs_elapsed.count() << " s" << std::endl;

  // compute C = Z.T * B
  FP* C = new FP[rank * rank]();
  gemm_start = std::chrono::high_resolution_clock::now();
  mklhelper<FP>::cblas_gemm(
      CblasRowMajor, // const CBLAS_LAYOUT Layout,
      CblasTrans,    // const CBLAS_TRANSPOSE transa,
      CblasNoTrans,  // const CBLAS_TRANSPOSE transb,
      rank,          // const MKL_INT m,
      rank,          // const MKL_INT n,
      n,             // const MKL_INT k,
      1.0,           // const float alpha,
      Z,             // const float *a,
      rank,          // const MKL_INT lda,
      B,             // const float *b,
      rank,          // const MKL_INT ldb,
      0.0,           // const float beta,
      C,             // float *c,
      rank           // const MKL_INT ldc
  );
  std::cout << "C = Z^T * B done, Range(C) = Range(B)." << std::endl;
  gemm_finish = std::chrono::high_resolution_clock::now();
  gemm_elapsed = gemm_finish - gemm_start;
  std::cout << "Elapsed time (gemm): " << gemm_elapsed.count() << " s" << std::endl;
  delete[] B;

  // C = U S V^T
  FP* U = new FP[rank * rank];
  S = new FP[rank];
  FP* superb = new FP[rank];
  auto gesvd_start = std::chrono::high_resolution_clock::now();
  lapack_status = mklhelper<FP>::LAPACKE_gesvd(
      LAPACK_ROW_MAJOR,     // int matrix_layout,
      'S',                  // char jobu,
      'N',                  // char jobvt,
      rank,                 // lapack_int m,
      rank,                 // lapack_int n,
      C,                    // float* a,
      rank,                 // lapack_int lda,
      S,                    // float* s,
      U,                    // float* u,
      rank,                 // lapack_int ldu,
      NULL,                 // float* vt,
      rank,                 // lapack_int ldvt,
      superb                // float* superb
    );
  assert(lapack_status == 0);
  std::cout << "JacabiSVD for C done." << std::endl;
  auto gesvd_finish = std::chrono::high_resolution_clock::now();
  std::chrono::duration<float> gesvd_elapsed = gesvd_finish - gesvd_start;
  std::cout << "Elapsed time (gesvd): " << gesvd_elapsed.count() << " s" << std::endl;
  delete[] C;

  // matU = Z * U
  matU = new FP[n * rank]();
  gemm_start = std::chrono::high_resolution_clock::now();
  mklhelper<FP>::cblas_gemm(
      CblasRowMajor, // const CBLAS_LAYOUT Layout,
      CblasNoTrans,  // const CBLAS_TRANSPOSE transa,
      CblasNoTrans,  // const CBLAS_TRANSPOSE transb,
      n,             // const MKL_INT m,
      rank,          // const MKL_INT n,
      rank,          // const MKL_INT k,
      1.0,           // const float alpha,
      Z,             // const float *a,
      rank,          // const MKL_INT lda,
      U,             // const float *b,
      rank,          // const MKL_INT ldb,
      0.0,           // const float beta,
      matU,          // float *c,
      rank           // const MKL_INT ldc
  );
  gemm_finish = std::chrono::high_resolution_clock::now();
  gemm_elapsed = gemm_finish - gemm_start;
  std::cout << "Elapsed time (gemm): " << gemm_elapsed.count() << " s" << std::endl;

  vslDeleteStream(&stream);
}
