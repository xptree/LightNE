#include "mklredsvd.h"
#include "util.h"
#include <chrono>  // for high_resolution_clock
#include <iostream>
#include <cassert>
#include <set>

#include "pbbslib/random.h"
#include "pbbslib/sequence_ops.h"

using namespace mkl_redsvd;

template <typename FP>
MKLRedSVD<FP>::MKLRedSVD(MKL_INT n, MKL_INT nnz, MKL_INT* row_idx, MKL_INT* col_idx, FP* value, bool upper, MKL_INT rank, bool analysis, bool random_project_only, bool sparse_project, float sparse_project_s) {
  std::cout << "init svd from coo" << std::endl;
  this->n = n;
  this->rank = rank;
  this->analysis = analysis;
  this->upper = upper;
  this->random_project_only = random_project_only;
  this->sparse_project = sparse_project;
  this->sparse_project_s = sparse_project_s;
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
MKLRedSVD<FP>::MKLRedSVD(MKL_INT n, MKL_INT* rows_start, MKL_INT* rows_end, MKL_INT* col_idx, FP* value, bool upper, MKL_INT rank, bool analysis, bool random_project_only, bool sparse_project, float sparse_project_s) {
  std::cout << "init svd from csr" << std::endl;
  this->n = n;
  this->rank = rank;
  this->analysis = analysis;
  this->upper = upper;
  this->random_project_only = random_project_only;
  this->sparse_project = sparse_project;
  this->sparse_project_s = sparse_project_s;
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
    mkl_free(S);
  }
  if (matU != NULL) {
    mkl_free(matU);
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
  MKL_INT expected_calls;
  if (random_project_only) {
    if (sparse_project) {
      expected_calls = 0;
    } else {
      expected_calls = 1;
    }
  } else {
    if (sparse_project) {
      expected_calls = 1;
    } else {
      expected_calls = 2;
    }
  }
  std::cout << "expected calls of spmm = " << expected_calls << std::endl; 
  if (analysis && expected_calls > 0) {
    auto start = std::chrono::high_resolution_clock::now();
    sparse_status_t status = mkl_sparse_set_mm_hint(
          csrA,                               // const sparse_matrix_t A,
          SPARSE_OPERATION_NON_TRANSPOSE,     // const sparse_operation_t operation,
          descrA,                             // const struct matrix_descr descr,
          SPARSE_LAYOUT_ROW_MAJOR,            // const sparse_layout_t layout,
          rank,                               // const MKL_INT dense_matrix_size,
          expected_calls                      // const MKL_INT expected_calls
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

  // compute Y = A.T * O = A * O, where O is the sample matrix.
  FP* Y = (FP*)mkl_calloc(n * rank, sizeof(FP), ALIGN);

  if (sparse_project) {
    // https://web.stanford.edu/~hastie/Papers/Ping/KDD06_rp.pdf
    FP s = sqrt(n);
    if (sparse_project_s >= 1.0) {
      s = static_cast<FP>(sparse_project_s);
    }
    FP density = 1. / s;
    std::cout << "very sparse random projection with s = " << s << " and density (1/s) = " << density << std::endl;
    int* non_zeros_per_row = new int[n];
    int err = viRngBinomial(VSL_RNG_METHOD_BINOMIAL_BTPE, //method
                        stream, //stream
                        n, //n
                        non_zeros_per_row, //r
                        rank, //n_trial
                        density // p
                        );
    assert(err == VSL_STATUS_OK);
    std::cout << "create #nnz per row done." << std::endl;
    MKL_INT* O_rows_start = new MKL_INT[n + 1];
    O_rows_start[0] = 0;
    for (size_t i=0; i<n; ++i) {
      O_rows_start[i+1] = O_rows_start[i] + non_zeros_per_row[i];
    }
    MKL_INT* O_col_idx = new MKL_INT[O_rows_start[n]];
    FP* O_value = new FP[O_rows_start[n]];
    std::cout << "#nnz = " << O_rows_start[n] << std::endl;
    auto seed = pbbs::random(time(0));

    int* tmp = new int[n * rank];
    parallel_for(0, n, [&] (MKL_INT i) {
      for (size_t j=0; j<rank; ++j) {
        tmp[i * rank + j] = j;
      }
      // Fisher–Yates_shuffle
      auto our_seed = seed.fork(i);
      for (size_t j=0; j<non_zeros_per_row[i]; ++j) {
        int k = j + our_seed.rand() % (rank - j);
        our_seed = our_seed.next();
        std::swap(tmp[i * rank + j], tmp[i * rank + k]);
        O_col_idx[O_rows_start[i] + j] = static_cast<MKL_INT>(tmp[i * rank + j]);
        O_value[O_rows_start[i] + j] = (our_seed.rand() % 2 == 0) ? -sqrt(s) : sqrt(s);
        our_seed = our_seed.next();
      }
      if (i == 0) {
        for (size_t j=0; j<non_zeros_per_row[i]; ++j) {
            std::cout << i << " " << O_col_idx[O_rows_start[i] + j] << " " << O_value[O_rows_start[i] + j] << std::endl;
        }
      }
    }, 1024);
    delete[] non_zeros_per_row;
    delete[] tmp;
    std::cout << "sampling very sparse random matrix O for A^T done, nnz= " << O_rows_start[n] << std::endl;
    sparse_matrix_t csrO;
    std::cout << "calling mkl_sparse_s_create_csr" << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    mkl_status = mklhelper<FP>::mkl_sparse_create_csr(&csrO,
        SPARSE_INDEX_BASE_ZERO,
        n,
        rank,
        O_rows_start,
        O_rows_start+1,
        O_col_idx,
        O_value);
    assert(mkl_status == SPARSE_STATUS_SUCCESS);
    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> elapsed = finish - start;
    std::cout << "Elapsed time: " << elapsed.count() << " s" << std::endl;
    std::cout << "create sparse csr matrix done, status=" << mkl_status << " n= " << n << " rank= " << rank << " nnz= " << O_rows_start[n] << std::endl;
    
    auto sp2md_start = std::chrono::high_resolution_clock::now();
	mkl_status = mklhelper<FP>::mkl_sparse_spmmd(SPARSE_OPERATION_NON_TRANSPOSE,csrA,csrO,SPARSE_LAYOUT_ROW_MAJOR,Y,rank);
    assert(mkl_status == SPARSE_STATUS_SUCCESS);
    std::cout << "compute sample matrix of Y = A^T * O  = A * O done (because A^T = A)" << std::endl;
    auto sp2md_finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> sp2md_elapsed = sp2md_finish - sp2md_start;
    std::cout << "Elapsed time (sp2md): " << sp2md_elapsed.count() << " s" << std::endl;
    mkl_status = mkl_sparse_destroy(csrO);
    std::cout << "destroy sparse projection matrix done, status=" << mkl_status << std::endl;
  } else {
    std::cout << "gaussian random matrix" << std::endl;
    // compute sample matrix O
    FP* O = (FP*)mkl_calloc(n * rank, sizeof(FP), ALIGN);
    util<FP>::standard_normal_vec_BM2(stream, n * rank, O);
    std::cout << "sampling gaussian random matrix O for A^T done." << std::endl;

    auto spmm_start = std::chrono::high_resolution_clock::now();
    std::cout << "going to call mkl_sparse_s_mm row major" << std::endl;
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
    mkl_free(O);
  }

  if (random_project_only) {
    std::cout << "only random random projected matrix." << std::endl;
    matU = Y;
    return;
  }
  // orthonormalize Y
  std::cout << "going to orthonormalize Y" << std::endl;
  auto gs_start = std::chrono::high_resolution_clock::now();
  util<FP>::gram_schmidt(n, rank, Y);
  std::cout << "orthonormalize Y done." << std::endl;
  auto gs_finish = std::chrono::high_resolution_clock::now();
  std::chrono::duration<float> gs_elapsed = gs_finish - gs_start;
  std::cout << "Elapsed time (gram schmidt): " << gs_elapsed.count() << " s" << std::endl;

  // compute B = A * Y;
  FP* B = (FP*)mkl_calloc(n * rank, sizeof(FP), ALIGN);
  auto spmm_start = std::chrono::high_resolution_clock::now();
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
  auto spmm_finish = std::chrono::high_resolution_clock::now();
  std::chrono::duration<float> spmm_elapsed = spmm_finish - spmm_start;
  std::cout << "Elapsed time (spmm): " << spmm_elapsed.count() << " s" << std::endl;
  mkl_sparse_destroy(csrA);

  // sample gaussian random matrix P
  FP* P = (FP*)mkl_calloc(rank * rank, sizeof(FP), ALIGN);
  util<FP>::standard_normal_vec_BM2(stream, rank * rank, P);
  std::cout << "sample another gaussian random matrix P done." << std::endl;

  FP* Z = (FP*)mkl_calloc(n * rank, sizeof(FP), ALIGN);
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
  mkl_free(P);

  // orthonormalize Z
  gs_start = std::chrono::high_resolution_clock::now();
  util<FP>::gram_schmidt(n, rank, Z);
  std::cout << "orthonormalize Z done." << std::endl;
  gs_finish = std::chrono::high_resolution_clock::now();
  gs_elapsed = gs_finish - gs_start;
  std::cout << "Elapsed time (gram schmidt): " << gs_elapsed.count() << " s" << std::endl;

  // compute C = Z.T * B
  FP* C = (FP*)mkl_calloc(rank * rank, sizeof(FP), ALIGN);
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
  mkl_free(B);

  // C = U S V^T
  FP* U = (FP*)mkl_calloc(rank * rank, sizeof(FP), ALIGN);
  S = (FP*)mkl_calloc(rank, sizeof(FP), ALIGN);
  FP* superb = (FP*)mkl_calloc(rank, sizeof(FP), ALIGN);
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
  mkl_free(C);

  // matU = Z * U
  matU = (FP*)mkl_calloc(n * rank, sizeof(FP), ALIGN);
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
