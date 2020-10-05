#pragma once
#include "mkl.h"

template <typename T>
struct mklhelper {};

template <>
struct mklhelper<float> {
  static sparse_status_t mkl_sparse_create_coo(sparse_matrix_t *A, const sparse_index_base_t indexing, const MKL_INT rows, const MKL_INT cols, const MKL_INT nnz, MKL_INT *row_indx, MKL_INT *col_indx, float *values) {
    return mkl_sparse_s_create_coo(A, indexing, rows, cols, nnz, row_indx, col_indx, values);
  }
  static sparse_status_t mkl_sparse_create_csr(sparse_matrix_t *A, const sparse_index_base_t indexing, const MKL_INT rows, const MKL_INT cols, MKL_INT *rows_start, MKL_INT *rows_end, MKL_INT *col_indx, float *values) {
    return mkl_sparse_s_create_csr(A, indexing, rows, cols, rows_start, rows_end, col_indx, values);
  }
  static int vRngGaussian(const MKL_INT method, VSLStreamStatePtr stream, const MKL_INT n, float* r, const float a, const float sigma) {
    return vsRngGaussian(method,
      stream,
      n,
      r,
      a,
      sigma
      );
  }
  static lapack_int LAPACKE_geqrf(int matrix_layout, lapack_int m, lapack_int n, float* a, lapack_int lda, float* tau) {
    return LAPACKE_sgeqrf(matrix_layout, m, n, a, lda, tau);
  }
  static lapack_int LAPACKE_orgqr(int matrix_layout, lapack_int m, lapack_int n, lapack_int k, float* a, lapack_int lda, const float* tau) {
    return LAPACKE_sorgqr(matrix_layout, m, n, k, a, lda, tau);
  }
  static sparse_status_t mkl_sparse_mm(const sparse_operation_t operation, const float alpha, const sparse_matrix_t A, const struct matrix_descr descr, const sparse_layout_t layout, const float *x, const MKL_INT columns, const MKL_INT ldx, const float beta, float *y, const MKL_INT ldy) {
    return mkl_sparse_s_mm(operation, alpha, A, descr, layout, x, columns, ldx, beta, y, ldy);
  }
  static void cblas_gemm(const CBLAS_LAYOUT Layout, const CBLAS_TRANSPOSE transa, const CBLAS_TRANSPOSE transb, const MKL_INT m, const MKL_INT n, const MKL_INT k, const float alpha, const float *a, const MKL_INT lda, const float *b, const MKL_INT ldb, const float beta, float *c, const MKL_INT ldc) {
    cblas_sgemm(Layout, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
  }
  static lapack_int LAPACKE_gesvd(int matrix_layout, char jobu, char jobvt, lapack_int m, lapack_int n, float* a, lapack_int lda, float* s, float* u, lapack_int ldu, float* vt, lapack_int ldvt, float* superb ) {
    return LAPACKE_sgesvd(matrix_layout, jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, superb );
  }
  static void cblas_axpy(const MKL_INT n, const float a, const float *x, const MKL_INT incx, float *y, const MKL_INT incy) {
    cblas_saxpy(n, a, x, incx, y, incy);
  }

  static void cblas_copy(const MKL_INT n, const float *x, const MKL_INT incx, float *y, const MKL_INT incy) {
    cblas_scopy(n, x, incx, y, incy);
  }

  static float cblas_nrm2(const MKL_INT n, const float *x, const MKL_INT incx) {
    return cblas_snrm2(n, x, incx);
  }

  static void cblas_scal(const MKL_INT n, const float a, float *x, const MKL_INT incx) {
    cblas_sscal(n, a, x, incx);
  }
};

template <>
struct mklhelper<double> {
  static sparse_status_t mkl_sparse_create_coo(sparse_matrix_t *A, const sparse_index_base_t indexing, const MKL_INT rows, const MKL_INT cols, const MKL_INT nnz, MKL_INT *row_indx, MKL_INT *col_indx, double *values) {
    return mkl_sparse_d_create_coo(A, indexing, rows, cols, nnz, row_indx, col_indx, values);
  }

  static sparse_status_t mkl_sparse_create_csr(sparse_matrix_t *A, const sparse_index_base_t indexing, const MKL_INT rows, const MKL_INT cols, MKL_INT *rows_start, MKL_INT *rows_end, MKL_INT *col_indx, double *values) {
    return mkl_sparse_d_create_csr(A, indexing, rows, cols, rows_start, rows_end, col_indx, values);
  }
  static int vRngGaussian(const MKL_INT method, VSLStreamStatePtr stream, const MKL_INT n, double* r, const double a, const double sigma) {
    return vdRngGaussian(method,
      stream,
      n,
      r,
      a,
      sigma
      );
  }
  static lapack_int LAPACKE_geqrf(int matrix_layout, lapack_int m, lapack_int n, double* a, lapack_int lda, double* tau) {
    return LAPACKE_dgeqrf(matrix_layout, m, n, a, lda, tau);
  }

  static lapack_int LAPACKE_orgqr(int matrix_layout, lapack_int m, lapack_int n, lapack_int k, double* a, lapack_int lda, const double* tau) {
    return LAPACKE_dorgqr(matrix_layout, m, n, k, a, lda, tau);
  }

  static sparse_status_t mkl_sparse_mm(const sparse_operation_t operation, const double alpha, const sparse_matrix_t A, const struct matrix_descr descr, const sparse_layout_t layout, const double *x, const MKL_INT columns, const MKL_INT ldx, const double beta, double *y, const MKL_INT ldy) {
    return mkl_sparse_d_mm(operation, alpha, A, descr, layout, x, columns, ldx, beta, y, ldy);
  }

  static void cblas_gemm(const CBLAS_LAYOUT Layout, const CBLAS_TRANSPOSE transa, const CBLAS_TRANSPOSE transb, const MKL_INT m, const MKL_INT n, const MKL_INT k, const double alpha, const double *a, const MKL_INT lda, const double *b, const MKL_INT ldb, const double beta, double *c, const MKL_INT ldc) {
    cblas_dgemm(Layout, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
  }

  static lapack_int LAPACKE_gesvd(int matrix_layout, char jobu, char jobvt, lapack_int m, lapack_int n, double* a, lapack_int lda, double* s, double* u, lapack_int ldu, double* vt, lapack_int ldvt, double* superb ) {
    return LAPACKE_dgesvd(matrix_layout, jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, superb );
  }

  static void cblas_axpy(const MKL_INT n, const double a, const double *x, const MKL_INT incx, double *y, const MKL_INT incy) {
    cblas_daxpy(n, a, x, incx, y, incy);
  }

  static void cblas_copy(const MKL_INT n, const double *x, const MKL_INT incx, double *y, const MKL_INT incy) {
    cblas_dcopy(n, x, incx, y, incy);
  }

  static double cblas_nrm2(const MKL_INT n, const double *x, const MKL_INT incx) {
    return cblas_dnrm2(n, x, incx);
  }

  static void cblas_scal(const MKL_INT n, const double a, double *x, const MKL_INT incx) {
    cblas_dscal(n, a, x, incx);
  }
};
