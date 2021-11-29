#pragma once
#include "mkl.h"
#include "mklhelper.h"

namespace mkl_util {

template <typename T>
struct mat{
  MKL_INT nrows, ncols;
  T *d;
};

template <typename T>
struct vec{
  MKL_INT vsize;
  T *d;
};

template <typename T>
struct mat_coo{
    MKL_INT nrows, ncols;
    long long nnz;      // number of non-zero element in the matrix.
    long long capacity; // number of possible nnzs.
    T *values;
    MKL_INT *rows, *cols;
};

template <typename T>
struct mat_csr{
    long long nnz;
    MKL_INT nrows, ncols;
    T *values;
    MKL_INT *cols;
    MKL_INT *pointerB, *pointerE;
};

template <typename T>
struct mat_csc{
    long long nnz;
    MKL_INT nrows, ncols;
    T *values;
    MKL_INT *rows;
    MKL_INT *pointerB, *pointerE;
};


template <typename T>
struct util {
  
  static mat<T> *matrix_new(MKL_INT nrows,MKL_INT ncols);

  static vec<T> *vector_new(MKL_INT vsize);

  static void matrix_delete(mat<T> *M);

  static void vector_delete(vec<T> *v);

  //row major format
  static void matrix_set_element(mat<T> *M, MKL_INT row_num, MKL_INT col_num, T val);

  static T matrix_get_element(mat<T> *M, MKL_INT row_num, MKL_INT col_num);

  static void vector_set_element(vec<T> *v, MKL_INT row_num, T val);

  static T vector_get_element(vec<T> *v, MKL_INT row_num);

  /* scale vector by a constant */
  static void vector_scale(vec<T> *v, T scalar);

  /* scale matrix by a constant */
  static void matrix_scale(mat<T> *M, T scalar);

  /* copy contents of vec s to d  */
  static void vector_copy(vec<T> *s,vec<T> *d);

  /* copy contents of mat S to D  */
  static void matrix_copy(mat<T> *S,mat<T> *D);

  //D = alpha*S
  static void scale_copy_matrix(mat<T> *S,mat<T> *D,T alpha);

  /* build transpose of matrix : Mt = M^T */
  static void matrix_build_transpose(mat<T> *M, mat<T> *Mt);

  /* subtract B from A and save result in A  */
  static void matrix_sub(mat<T> *A, mat<T> *B);

  /* initialize diagonal matrix from vector data */
  static void initialize_diagonal_matrix(mat<T> *D, vec<T> *data);

  /* initialize identity */
  static void initialize_identity_matrix(mat<T> *D);

  static void matrix_matrix_mult(mat<T> *A, mat<T> *B, mat<T> *C,T alpha,T beta);

  static void matrix_transpose_matrix_mult(mat<T> *A, mat<T> *B, mat<T> *C,T alpha,T beta);

  static void matrix_matrix_transpose_mult(mat<T> *A,mat<T> *B,mat<T> *C,T alpha,T beta);

  static void matrix_transpose_matrix_transpose_mult(mat<T> *A, mat<T> *B, mat<T> *C,T alpha,T beta);

  /* set column of matrix to vector */
  static void matrix_set_col(mat<T> *M, MKL_INT j, vec<T> *column_vec);

  /* extract column of a matrix into a vector */
  static void matrix_get_col(mat<T> *M, MKL_INT j, vec<T> *column_vec);

  /* extract row i of a matrix into a vector */
  static void matrix_get_row(mat<T> *M, MKL_INT i, vec<T> *row_vec);

  /* put vector row_vec as row i of a matrix */
  static void matrix_set_row(mat<T> *M, MKL_INT i, vec<T> *row_vec);

  /* Mc = M(:,inds) */
  static void matrix_get_selected_columns(mat<T> *M, MKL_INT *inds, mat<T> *Mc);

  /* M(:,inds) = Mc */
  static void matrix_set_selected_columns(mat<T> *M, MKL_INT *inds, mat<T> *Mc);

  /* Mr = M(inds,:) */
  static void matrix_get_selected_rows(mat<T> *M, MKL_INT *inds, mat<T> *Mr);

  /* M(inds,:) = Mr */
  static void matrix_set_selected_rows(mat<T> *M, MKL_INT *inds, mat<T> *Mr);
  
  //B = alpha*A+B
  static void matrix_matrix_add(mat<T> *A, mat<T> *B,T alpha);

  //B = (diag)S*A+B; A input is a matrix(n,1)
  static void diag_matrix_mult(mat<T> *S, mat<T> *A, mat<T> *B);

  //B = A*(diag)S+B; A input is a matrix(n,1)
  static void matrix_diag_mult(mat<T> *A, mat<T> *S, mat<T> *B);

  /* Performs [Q,R] = qr(M,'0') compact QR factorization 
  M is mxn ; Q is mxn ; R is min(m,n) x min(m,n) */
  static void compact_QR_factorization(mat<T> *M, mat<T> *Q, mat<T> *R);

  /* returns Q from [Q,R] = qr(M,'0') compact QR factorization 
  M is mxn ; Q is mxn ; R is min(m,n) x min(m,n) */
  static void QR_factorization_getQ(mat<T> *M, mat<T> *Q);

  static void QR_factorization_getQ_inplace(mat<T> *Q);

  /* computes SVD: M = U*S*Vt; note Vt = V^T */
  static void singular_value_decomposition(mat<T> *M, mat<T> *U, mat<T> *S, mat<T> *Vt);

  static void initialize_random_gaussian_matrix(mat<T> *M);

  static void LUfraction(mat<T> *L);

  /*[U, S, V] = eigSVD(A)*/
  static void eigSVD(mat<T> *A, mat<T> *U, mat<T> *S, mat<T> *V);

  //sparse matrix operation:

  static mat_coo<T> *coo_matrix_new(MKL_INT nrows, MKL_INT ncols, long long capacity);

  static void coo_matrix_delete(mat_coo<T> *M);

  static mat_csr<T> *csr_matrix_new(MKL_INT nrows, MKL_INT ncols, long long capacity);

  static mat_csr<T> *csr_new_from_coo(mat_coo<T> *COO);

  static mat_csr<T> *csr_new_from_csr(mat_csr<T> *src);

  //mat M is a size(n,1) matrix
  static mat_csr<T> *csr_new_from_diag(mat<T> *M);

  static void sparse_csr_mm(sparse_matrix_t csrA,mat<T> *M,mat<T> *Y,T alpha,T beta,struct matrix_descr descrM);

  static void sparse_csr_t_mm(sparse_matrix_t csrA,mat<T> *M,mat<T> *Y,T alpha,T beta,struct matrix_descr descrM);

  static void sparse_csr_spmmd(sparse_matrix_t csrA,sparse_matrix_t csrB,mat<T> *C);

  static void sparse_csr_spm2d(sparse_matrix_t csrA,sparse_matrix_t csrB,mat<T> *C,T alpha,T beta,struct matrix_descr descrA,struct matrix_descr descrB);

  static void csr_matrix_matrix_mult(mat_csr<T> *A, mat<T> *B, mat<T> *C,T alpha,T beta);
  
  static void csr_matrix_delete(mat_csr<T> *M);

  static mat_csc<T> *csc_matrix_new(MKL_INT nrows, MKL_INT ncols, long long capacity);

  static void csc_matrix_delete(mat_csc<T> *M);

};

} // namespace mkl_util
#include "mklutil.cpp"
