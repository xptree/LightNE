
#include "mklutil.h"
#include <cassert>
#include <cmath>
#include <iostream>
#define ALIGN (64)

namespace mkl_util{

template <typename T>
mat<T> *util<T>::matrix_new(MKL_INT nrows,MKL_INT ncols){
  mat<T> *M = new mat<T>;
  MKL_INT size_mat = nrows*ncols;
  //M->d = new T[size_mat]();
  //M->d = (T*) mkl_calloc(size_mat,sizeof(T),ALIGN);
  M->d = (T*) mkl_malloc(size_mat*sizeof(T),ALIGN);
  M->nrows = nrows;
  M->ncols = ncols;
  return M;
}

template <typename T>
vec<T> * util<T>::vector_new(MKL_INT vsize){
  vec<T> *v = new vec<T>;
  //v->d = new T[vsize]();
  v->d = (T*) mkl_calloc(vsize,sizeof(T),ALIGN);
  v->vsize = vsize;
  return v;
}

template <typename T>
void util<T>::matrix_delete(mat<T> *M){
  if (M!=NULL){
    //delete M->d;
    mkl_free(M->d);
    delete M;
    M = NULL;
  }
}

template <typename T>
void util<T>::vector_delete(vec<T> *v){
  if (v!=NULL){
    //delete v->d;
    mkl_free(v->d);
    delete v;
    v = NULL;
  }
}

//row major format
template <typename T>
void util<T>::matrix_set_element(mat<T> *M, MKL_INT row_num, MKL_INT col_num, T val){
  M->d[M->ncols*row_num+col_num] = val;
}

template <typename T>
T util<T>::matrix_get_element(mat<T> *M, MKL_INT row_num, MKL_INT col_num){
  return M->d[M->ncols*row_num+col_num];
}

template <typename T>
void util<T>::vector_set_element(vec<T> *v, MKL_INT row_num, T val){
  v->d[row_num] = val;
}

template <typename T>
T util<T>::vector_get_element(vec<T> *v, MKL_INT row_num){
  return v->d[row_num];
}

/* scale vector by a constant */
template <typename T>
void util<T>::vector_scale(vec<T> *v, T scalar)
{
  MKL_INT i;
  mklhelper<T>::cblas_scal((v->vsize),scalar,v->d,1);
}

/* scale matrix by a constant */
template <typename T>
void util<T>::matrix_scale(mat<T> *M, T scalar)
{
  mklhelper<T>::cblas_scal((M->nrows) * (M->ncols),scalar,M->d,1);
}

/* copy contents of vec s to d  */
template <typename T>
void util<T>::vector_copy(vec<T> *s,vec<T> *d)
{
  MKL_INT i;
  mklhelper<T>::cblas_copy(s->vsize,s->d,1,d->d,1);
}

/* copy contents of mat S to D  */
template <typename T>
void util<T>::matrix_copy(mat<T> *S,mat<T> *D)
{
  MKL_INT n = S->nrows * S->ncols;
  mklhelper<T>::cblas_copy(n,S->d,1,D->d,1);
}

//D = alpha*S
template <typename T>
void util<T>::scale_copy_matrix(mat<T> *S,mat<T> *D,T alpha){
  MKL_INT i;
  #pragma omp parallel shared(D, S, alpha) private(i)
  {
  #pragma omp for
      for (i = 0; i < ((S->nrows) * (S->ncols)); i++)
      {
          D->d[i] = alpha*S->d[i];
      }
  }
}

/* build transpose of matrix : Mt = M^T */
template <typename T>
void util<T>::matrix_build_transpose(mat<T> *M, mat<T> *Mt){
  MKL_INT i, j;
  #pragma omp parallel shared(M,Mt) private(i,j)
  {
  #pragma omp for
      for (i = 0; i < (M->nrows); i++)
      {
          for (j = 0; j < (M->ncols); j++)
          {
              matrix_set_element(Mt, j, i, matrix_get_element(M, i, j));
          }
      }
  }
}

/* subtract B from A and save result in A  */
template <typename T>
void util<T>::matrix_sub(mat<T> *A, mat<T> *B)
{
  MKL_INT i;
  #pragma omp parallel shared(A, B) private(i)
  {
  #pragma omp for
      for (i = 0; i < ((A->nrows) * (A->ncols)); i++)
      {
          A->d[i] = A->d[i] - B->d[i];
      }
  }
}

/* initialize diagonal matrix from vector data */
template <typename T>
void util<T>::initialize_diagonal_matrix(mat<T> *D, vec<T> *data)
{
  MKL_INT i;
  #pragma omp parallel shared(D) private(i)
  {
  #pragma omp parallel for
      for (i = 0; i < (D->nrows); i++)
      {
          matrix_set_element(D, i, i, data->d[i]);
      }
  }
}

/* initialize identity */
template <typename T>
void util<T>::initialize_identity_matrix(mat<T> *D)
{
  MKL_INT i;
  #pragma omp parallel shared(D) private(i)
  {
  #pragma omp parallel for
      for (i = 0; i < (D->nrows); i++)
      {
          matrix_set_element(D, i, i, 1.0);
      }
  }
}

template <typename T>
void util<T>::matrix_matrix_mult(mat<T> *A, mat<T> *B, mat<T> *C,T alpha,T beta){
  mklhelper<T>::cblas_gemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, A->nrows, B->ncols, A->ncols, alpha, A->d, A->ncols, B->d, B->ncols, beta, C->d, C->ncols);
}

template <typename T>
void util<T>::matrix_transpose_matrix_mult(mat<T> *A, mat<T> *B, mat<T> *C,T alpha,T beta){
  mklhelper<T>::cblas_gemm(CblasRowMajor, CblasTrans, CblasNoTrans, A->ncols, B->ncols, A->nrows, alpha, A->d, A->ncols, B->d, B->ncols, beta, C->d, B->ncols);
}

template <typename T>
void util<T>::matrix_matrix_transpose_mult(mat<T> *A,mat<T> *B,mat<T> *C,T alpha,T beta){
  mklhelper<T>::cblas_gemm(CblasRowMajor, CblasNoTrans, CblasTrans, A->nrows, B->nrows, A->ncols, alpha, A->d, A->ncols, B->d, A->ncols, beta, C->d, B->nrows);
}

template <typename T>
void util<T>::matrix_transpose_matrix_transpose_mult(mat<T> *A, mat<T> *B, mat<T> *C,T alpha,T beta){
  mklhelper<T>::cblas_gemm(CblasRowMajor, CblasTrans, CblasTrans, A->ncols, B->nrows, A->nrows, alpha, A->d, A->ncols, B->d, A->nrows, beta, C->d, B->nrows);
}

/* set column of matrix to vector */
template <typename T>
void util<T>::matrix_set_col(mat<T> *M, MKL_INT j, vec<T> *column_vec)
{
  MKL_INT i;
  #pragma omp parallel shared(column_vec, M, j) private(i)
  {
  #pragma omp for
      for (i = 0; i < M->nrows; i++)
      {
          matrix_set_element(M, i, j, vector_get_element(column_vec, i));
      }
  }
}

/* extract column of a matrix into a vector */
template <typename T>
void util<T>::matrix_get_col(mat<T> *M, MKL_INT j, vec<T> *column_vec)
{
  MKL_INT i;
  #pragma omp parallel shared(column_vec, M, j) private(i)
  {
  #pragma omp for
      for (i = 0; i < M->nrows; i++)
      {
          vector_set_element(column_vec, i, matrix_get_element(M, i, j));
      }
  }
}

/* extract row i of a matrix into a vector */
template <typename T>
void util<T>::matrix_get_row(mat<T> *M, MKL_INT i, vec<T> *row_vec)
{
  MKL_INT j;
  #pragma omp parallel shared(row_vec, M, i) private(j)
  {
  #pragma omp for
      for (j = 0; j < M->ncols; j++)
      {
          vector_set_element(row_vec, j, matrix_get_element(M, i, j));
      }
  }
}

/* put vector row_vec as row i of a matrix */
template <typename T>
void util<T>::matrix_set_row(mat<T> *M, MKL_INT i, vec<T> *row_vec)
{
  MKL_INT j;
  #pragma omp parallel shared(row_vec, M, i) private(j)
  {
  #pragma omp for
      for (j = 0; j < M->ncols; j++)
      {
          matrix_set_element(M, i, j, vector_get_element(row_vec, j));
      }
  }
}

/* Mc = M(:,inds) */
template <typename T>
void util<T>::matrix_get_selected_columns(mat<T> *M, MKL_INT *inds, mat<T> *Mc)
{
  MKL_INT i;
  vec<T> *col_vec;
  #pragma omp parallel shared(M, Mc, inds) private(i, col_vec)
  {
  #pragma omp for
      for (i = 0; i < (Mc->ncols); i++)
      {
          col_vec = vector_new(M->nrows);
          matrix_get_col(M, inds[i], col_vec);
          matrix_set_col(Mc, i, col_vec);
          vector_delete(col_vec);
      }
  }
}

/* M(:,inds) = Mc */
template <typename T>
void util<T>::matrix_set_selected_columns(mat<T> *M, MKL_INT *inds, mat<T> *Mc)
{
  MKL_INT i;
  vec<T> *col_vec;
  #pragma omp parallel shared(M, Mc, inds) private(i, col_vec)
  {
  #pragma omp parallel for
      for (i = 0; i < (Mc->ncols); i++)
      {
          col_vec = vector_new(M->nrows);
          matrix_get_col(Mc, i, col_vec);
          matrix_set_col(M, inds[i], col_vec);
          vector_delete(col_vec);
      }
  }
}

/* Mr = M(inds,:) */
template <typename T>
void util<T>::matrix_get_selected_rows(mat<T> *M, MKL_INT *inds, mat<T> *Mr)
{
  MKL_INT i;
  vec<T> *row_vec;
  #pragma omp parallel shared(M, Mr, inds) private(i, row_vec)
  {
  #pragma omp parallel for
      for (i = 0; i < (Mr->nrows); i++)
      {
          row_vec = vector_new(M->ncols);
          matrix_get_row(M, inds[i], row_vec);
          matrix_set_row(Mr, i, row_vec);
          vector_delete(row_vec);
      }
  }
}

/* M(inds,:) = Mr */
template <typename T>
void util<T>::matrix_set_selected_rows(mat<T> *M, MKL_INT *inds, mat<T> *Mr)
{
  MKL_INT i;
  vec<T> *row_vec;
  #pragma omp parallel shared(M, Mr, inds) private(i, row_vec)
  {
  #pragma omp parallel for
      for (i = 0; i < (Mr->nrows); i++)
      {
          row_vec = vector_new(M->ncols);
          matrix_get_row(Mr, i, row_vec);
          matrix_set_row(M, inds[i], row_vec);
          vector_delete(row_vec);
      }
  }
}

//B = alpha*A+B
template <typename T>
void util<T>::matrix_matrix_add(mat<T> *A, mat<T> *B,T alpha)
{
  long long n = A->nrows;
  n = n * (A->ncols);
  MKL_INT incx = 1, incy = 1;
  mklhelper<T>::cblas_axpy(n, alpha, A->d, incx, B->d, incy);
  return;
}

//B = (diag)S*A+B; A input is a matrix(n,1)
template <typename T>
void util<T>::diag_matrix_mult(mat<T> *S, mat<T> *A, mat<T> *B)
{
  MKL_INT i;
  #pragma omp parallel shared(S,A,B) private(i)
  {
  #pragma omp for
    for (i = 0;i < A->nrows;i++){
      T num = matrix_get_element(S,i,0);
      mklhelper<T>::cblas_axpy(A->ncols, num, A->d+i*A->ncols, 1, B->d+i*A->ncols, 1);
    }
  }
  return;
}


//B = A*(diag)S+B; A input is a matrix(n,1)
template <typename T>
void util<T>::matrix_diag_mult(mat<T> *A, mat<T> *S, mat<T> *B)
{
  MKL_INT i;
  #pragma omp parallel shared(S,A,B) private(i)
  {
  #pragma omp for
    for (i = 0;i < A->ncols;i++){
      T num = matrix_get_element(S,i,0);
      mklhelper<T>::cblas_axpy(A->nrows, num, A->d+i, A->ncols, B->d+i, A->ncols);
    }
  }
  return;
}



/* Performs [Q,R] = qr(M,'0') compact QR factorization 
M is mxn ; Q is mxn ; R is min(m,n) x min(m,n) */
template <typename T>
void util<T>::compact_QR_factorization(mat<T> *M, mat<T> *Q, mat<T> *R)
{
  MKL_INT i, j, m, n, k;
  m = M->nrows;
  n = M->ncols;
  k = std::min(m, n);
  mat<T> *R_full = matrix_new(m, n);
  matrix_copy(M,R_full);

  vec<T> *tau = vector_new(k);
  printf("sgeqrf..\n");
  lapack_int info = mklhelper<T>::LAPACKE_geqrf(LAPACK_ROW_MAJOR, R_full->nrows, R_full->ncols, R_full->d, R_full->ncols, tau->d);
  if (info!=0){
    std::cout<<"some Error happen in the geqrf,info:"<<info<<std::endl;
  }

  for (i = 0; i < k; i++)
  {
      for (j = 0; j < k; j++)
      {
          if (j >= i)
          {
              matrix_set_element(R, i, j, matrix_get_element(R_full, i, j));
          }
      }
  }

  // get Q
  matrix_copy(R_full,Q);
  printf("sorgqr..\n");
  info = mklhelper<T>::LAPACKE_orgqr(LAPACK_ROW_MAJOR, Q->nrows, Q->ncols, std::min(Q->ncols, Q->nrows), Q->d, Q->ncols, tau->d);
  if (info!=0){
    std::cout<<"some Error happen in the orgqr,info:"<<info<<std::endl;
  }
  // clean up
  matrix_delete(R_full);
  vector_delete(tau);
}

/* returns Q from [Q,R] = qr(M,'0') compact QR factorization 
M is mxn ; Q is mxn ; R is min(m,n) x min(m,n) */
template <typename T>
void util<T>::QR_factorization_getQ(mat<T> *M, mat<T> *Q)
{
  MKL_INT m, n, k;
  m = M->nrows;
  n = M->ncols;
  k = std::min(m, n);
  matrix_copy(M,Q);
  vec<T> *tau = vector_new(k);
  printf("sgeqrf..\n");
  lapack_int info = mklhelper<T>::LAPACKE_geqrf(LAPACK_ROW_MAJOR, m, n, Q->d, n, tau->d);
  if (info!=0){
    std::cout<<"some Error happen in the geqrf,info:"<<info<<std::endl;
  }
  printf("sorgqr..\n");
  info = mklhelper<T>::LAPACKE_orgqr(LAPACK_ROW_MAJOR, m, n, n, Q->d, n, tau->d);
  if (info!=0){
    std::cout<<"some Error happen in the orgqr,info:"<<info<<std::endl;
  }

  // clean up
  vector_delete(tau);
}

template <typename T>
void util<T>::QR_factorization_getQ_inplace(mat<T> *Q)
{
  MKL_INT m, n, k;
  m = Q->nrows;
  n = Q->ncols;
  k = std::min(m, n);
  
  MKL_INT *jpvt = new MKL_INT[n];;
  vec<T> *tau = vector_new(k);
  printf("sgeqrf..\n");
  lapack_int info = mklhelper<T>::LAPACKE_geqpf(LAPACK_ROW_MAJOR, m, n, Q->d, n, jpvt, tau->d);
  if (info!=0){
    std::cout<<"some Error happen in the geqpf,info:"<<info<<std::endl;
  }
  printf("sorgqr..\n");
  info = mklhelper<T>::LAPACKE_orgqr(LAPACK_ROW_MAJOR, m, n, n, Q->d, n, tau->d);
  if (info!=0){
    std::cout<<"some Error happen in the orgqr,info:"<<info<<std::endl;
  }
  // clean up
  vector_delete(tau);
  delete jpvt;
}

/* computes SVD: M = U*S*Vt; note Vt = V^T */
template <typename T>
void util<T>::singular_value_decomposition(mat<T> *M, mat<T> *U, mat<T> *S, mat<T> *Vt)
{
  MKL_INT m, n, k;
  m = M->nrows;
  n = M->ncols;
  k = std::min(m, n);
  vec<T> *work = vector_new(2 * std::max(3 * std::min(m, n) + std::max(m, n), 5 * std::min(m, n)));
  vec<T> *svals = vector_new(k);

  lapack_int info = mklhelper<T>::LAPACKE_gesvd(LAPACK_ROW_MAJOR, 'S', 'S', m, n, M->d, n, svals->d, U->d, k, Vt->d, n, work->d);
  //lapack_int info = mklhelper<T>::LAPACKE_gesdd(LAPACK_ROW_MAJOR,'S',m,n,M->d,n,svals->d,U->d,n,Vt->d,n);//m>=n,or m<n need different parameter
  if (info!=0){
    std::cout<<"some Error happen in the gesvd(gesdd),info:"<<info<<std::endl;
  }
  initialize_diagonal_matrix(S, svals);

  vector_delete(work);
  vector_delete(svals);
}

template <typename T>
void util<T>::initialize_random_gaussian_matrix(mat<T> *M){
  MKL_INT m, n;
  m = M->nrows;
  n = M->ncols;
  T a = 0.0, sigma = 1.0;
  MKL_INT N = m * n;
  VSLStreamStatePtr stream;
  //vslNewStream(&stream, BRNG, time(NULL));
  vslNewStream( &stream, BRNG,  SEED );
  int err = mklhelper<T>::vRngGaussian(METHOD, stream, N, M->d, a, sigma);
  assert(err == VSL_STATUS_OK);
  std::cout<<"building gaussian random matrix finished"<<std::endl;
}

/*[L, ~] = lu(A) as in MATLAB*/
template <typename T>
void util<T>::LUfraction(mat<T> *L)
{
  //matrix_copy(A, L);
  lapack_int *ipiv = new lapack_int[L->ncols];
  MKL_INT info = mklhelper<T>::LAPACKE_getrf(LAPACK_ROW_MAJOR, L->nrows, L->ncols, L->d, L->ncols, ipiv);    
  if (info!=0)
  {
    std::cout<<"some Error happen in the getrf,info:"<<info<<std::endl;
    exit(1);
  }

  MKL_INT i,j;
  #pragma omp parallel shared(L) private(i,j) 
  {
  #pragma omp for     
      for(i=0;i<L->ncols;i++)
      {
          for(j=0;j<i;j++)
          {
              matrix_set_element(L,j,i,0);
          }
          matrix_set_element(L,i,i,1);
      }
  }
  
  {
      for(i=L->ncols-1;i>=0;i--)
      {
          lapack_int ipi = ipiv[i]-1;
          #pragma omp parallel shared(L,ipi) private(j) 
          {
          #pragma omp for
          for(j=0;j<L->ncols;j++)
          {
              T temp = matrix_get_element(L,ipi,j);
              matrix_set_element(L,ipi,j,matrix_get_element(L,i,j));
              matrix_set_element(L,i,j,temp);
          }
          }
      }
  }
  delete ipiv;
}

/*[U, S, V] = eigSVD(A),A is a n*k matrix,U rewrite to A*/
template <typename T>
void util<T>::eigSVD(mat<T> *A, mat<T> *U, mat<T> *S, mat<T> *V)
{
  matrix_transpose_matrix_mult(A, A, V, 1.0, 0.0);
  //lapack_int info = mklhelper<T>::LAPACKE_syevd(LAPACK_ROW_MAJOR, 'V', 'U', V->ncols, V->d, V->ncols, S->d);//some bug in floating number
  lapack_int info = mklhelper<T>::LAPACKE_syev(LAPACK_ROW_MAJOR,'V','U',V->ncols,V->d,V->ncols,S->d);
  if (info!=0)
  {
    std::cout<<"some Error happen in the eigSVD syev(syevd),info:"<<info<<std::endl;
    exit(1);
  }
  mat<T> *V1 = matrix_new(V->ncols, V->ncols);
  matrix_copy(V,V1);
  MKL_INT i, j;
  #pragma omp parallel shared(V1,S) private(i,j) 
  {
  #pragma omp for
      for(i=0; i<V1->ncols; i++)
      {
          S->d[i] = sqrt(S->d[i]);
          if (S->d[i]==0){
            std::cout<<"there is numerical problem when computing eigSVD, recommend to use QR(SVD)"<<std::endl;
            exit(1);
          }
          for(j=0; j<V1->nrows;j++)
              matrix_set_element(V1,j,i,matrix_get_element(V1,j,i)/S->d[i]);
      }
  }
  matrix_matrix_mult(A, V1, U, 1.0, 0);
  matrix_delete(V1);
}

template <typename T>
mat_coo<T> *util<T>::coo_matrix_new(MKL_INT nrows, MKL_INT ncols, long long capacity)
{
  mat_coo<T> *M = new mat_coo<T>;
  //M->values = new T[capacity];
  //M->rows = new MKL_INT[capacity];
  //M->cols = new MKL_INT[capacity];
  M->values = (T*) mkl_malloc(capacity*sizeof(T),ALIGN);
  M->rows = (MKL_INT*) mkl_malloc(capacity*sizeof(MKL_INT),ALIGN);
  M->cols = (MKL_INT*) mkl_malloc(capacity*sizeof(MKL_INT),ALIGN);
  M->nnz = capacity;
  M->nrows = nrows;
  M->ncols = ncols;
  M->capacity = capacity;
  return M;
}

template <typename T>
void util<T>::coo_matrix_delete(mat_coo<T> *M)
{
  if (M!=NULL){
    // delete M->values;
    // delete M->cols;
    // delete M->rows;
    mkl_free(M->values);
    mkl_free(M->cols);
    mkl_free(M->rows);
    delete M;
    M = NULL;
  }
}


template <typename T>
mat_csr<T> *util<T>::csr_matrix_new(MKL_INT nrows, MKL_INT ncols, long long capacity)
{
  mat_csr<T> *M = new mat_csr<T>;
  // M->pointerB = new MKL_INT[(nrows + 1)]();
  // M->pointerE = new MKL_INT[(nrows + 1)]();
  // M->cols = new MKL_INT[capacity];
  // M->values = new T[capacity];
  M->values = (T*) mkl_malloc(capacity*sizeof(T),ALIGN);
  M->cols = (MKL_INT*) mkl_malloc(capacity*sizeof(MKL_INT),ALIGN);
  M->pointerB = (MKL_INT*) mkl_malloc((nrows+1)*sizeof(MKL_INT),ALIGN);
  M->pointerE = M->pointerB + 1;
  M->nnz = capacity;
  M->nrows = nrows;
  M->ncols = ncols;
  return M;
}


template <typename T>
mat_csr<T> *util<T>::csr_new_from_coo(mat_coo<T> *COO)
{
  mat_csr<T> *CSR = new mat_csr<T>;
  CSR->nrows = COO->nrows;
  CSR->ncols = COO->ncols;
  CSR->nnz = COO->nnz;
  // CSR->pointerB = new MKL_INT[(CSR->nrows+1)]();
  // CSR->pointerE = new MKL_INT[(CSR->nrows+1)]();
  // CSR->cols = new MKL_INT[CSR->nnz];
  // CSR->values = new T[CSR->nnz];
  CSR->values = (T*) mkl_malloc(CSR->nnz*sizeof(T),ALIGN);
  CSR->cols = (MKL_INT*) mkl_malloc(CSR->nnz*sizeof(MKL_INT),ALIGN);
  CSR->pointerB = (MKL_INT*) mkl_malloc((CSR->nrows+1)*sizeof(MKL_INT),ALIGN);
  CSR->pointerE = CSR->pointerB + 1;

  mklhelper<T>::cblas_copy(COO->nnz,COO->values,1,CSR->values,1);
  MKL_INT current_row, cursor = 0;
  for (current_row = 0; current_row < CSR->nrows; current_row++)
  {
      CSR->pointerB[current_row] = cursor;
      while (COO->rows[cursor] == current_row)
      {
          CSR->cols[cursor] = COO->cols[cursor];
          cursor++;
      }
      //CSR->pointerE[current_row] = cursor;
  }
  CSR->pointerB[CSR->nrows] = cursor;
  return CSR;
}

template <typename T>
mat_csr<T> *util<T>::csr_new_from_csr(mat_csr<T> *src)
{
  mat_csr<T> *dest = new mat_csr<T>;
  dest->nrows = src->nrows;
  dest->ncols = src->ncols;
  dest->nnz = src->nnz;
  // dest->pointerB = new MKL_INT[(dest->nrows+1)]();
  // dest->pointerE = new MKL_INT[(dest->nrows+1)]();
  // dest->cols = new MKL_INT[dest->nnz];
  // dest->values = new T[dest->nnz];
  dest->values = (T*) mkl_malloc(dest->nnz*sizeof(T),ALIGN);
  dest->cols = (MKL_INT*) mkl_malloc(dest->nnz*sizeof(MKL_INT),ALIGN);
  dest->pointerB = (MKL_INT*) mkl_malloc((dest->nrows+1)*sizeof(MKL_INT),ALIGN);
  dest->pointerE = dest->pointerB + 1;
  mklhelper<T>::cblas_copy(src->nnz,src->values,1,dest->values,1);
  memcpy(dest->cols,src->cols,src->nnz * sizeof(MKL_INT));
  memcpy(dest->pointerB,src->pointerB,(dest->nrows+1)*sizeof(MKL_INT));
  return dest;
}

//mat M is a size(n,1) matrix
template <typename T>
mat_csr<T> *util<T>::csr_new_from_diag(mat<T> *M)
{
  mat_csr<T> *A = new mat_csr<T>;
  A->nrows = M->nrows;
  A->ncols = M->nrows;
  A->nnz = M->nrows;
  // A->pointerB = new MKL_INT[(A->nrows+1)]();
  // A->pointerE = new MKL_INT[(A->nrows+1)]();
  // A->cols = new MKL_INT[A->nrows];
  // A->values = new T[A->nrows];
  A->values = (T*) mkl_malloc(A->nrows*sizeof(T),ALIGN);
  A->cols = (MKL_INT*) mkl_malloc(A->nrows*sizeof(MKL_INT),ALIGN);
  A->pointerB = (MKL_INT*) mkl_malloc((A->nrows+1)*sizeof(MKL_INT),ALIGN);
  A->pointerE = A->pointerB + 1;
  mklhelper<T>::cblas_copy(M->nrows,M->d,1,A->values,1);

  MKL_INT current_row, cursor = 0;
  for (current_row = 0; current_row < A->nrows; current_row++)
  {
    if (M->d[current_row]!=0){
      A->cols[current_row] = current_row;
      A->pointerB[current_row] = cursor;
      cursor++;
      //A->pointerE[current_row] = cursor;
    }
  }
  A->pointerB[current_row] = cursor;
  return A;
}


template <typename T>
void util<T>::sparse_csr_mm(sparse_matrix_t csrA,mat<T> *M,mat<T> *Y,T alpha,T beta,struct matrix_descr descrM){
  MKL_INT l = M->ncols;
  sparse_status_t mkl_status = mklhelper<T>::mkl_sparse_mm(SPARSE_OPERATION_NON_TRANSPOSE,
                  alpha,
                  csrA,
                  descrM,
                  SPARSE_LAYOUT_ROW_MAJOR,
                  M->d,
                  l,   // number of right hand sides
                  l,      // ldx
                  beta,
                  Y->d,
                  l);
  if (mkl_status != SPARSE_STATUS_SUCCESS){
      printf("mkl_sparse_csr_mm error\n");
  }
}

template <typename T>
void util<T>::sparse_csr_t_mm(sparse_matrix_t csrA,mat<T> *M,mat<T> *Y,T alpha,T beta,struct matrix_descr descrM){
  MKL_INT l = M->ncols;
  sparse_status_t mkl_status = mklhelper<T>::mkl_sparse_mm(SPARSE_OPERATION_TRANSPOSE,
                  alpha,
                  csrA,
                  descrM,
                  SPARSE_LAYOUT_ROW_MAJOR,
                  M->d,
                  l,   // number of right hand sides
                  l,      // ldx
                  beta,
                  Y->d,
                  l);
  if (mkl_status != SPARSE_STATUS_SUCCESS){
      printf("mkl_sparse_csr_t_mm error\n");
  }
}

template <typename T>
void util<T>::sparse_csr_spmmd(sparse_matrix_t csrA,sparse_matrix_t csrB,mat<T> *C){
  MKL_INT l = C->ncols;
  sparse_status_t mkl_status = mklhelper<T>::mkl_sparse_spmmd(SPARSE_OPERATION_NON_TRANSPOSE,csrA,
                  csrB,
                  SPARSE_LAYOUT_ROW_MAJOR,
                  C->d,
                  l);
  if (mkl_status != SPARSE_STATUS_SUCCESS){
      printf("mkl_sparse_spmmd error stauts:%d\n",mkl_status);
  }
}


template <typename T>
void util<T>::sparse_csr_spm2d(sparse_matrix_t csrA,sparse_matrix_t csrB,mat<T> *C,T alpha,T beta,struct matrix_descr descrA,struct matrix_descr descrB){
  MKL_INT n = C->nrows;
  sparse_status_t mkl_status = mklhelper<T>::mkl_sparse_sp2md(SPARSE_OPERATION_NON_TRANSPOSE,descrA,csrA,
                  SPARSE_OPERATION_NON_TRANSPOSE,descrB,csrB,
                  alpha,beta,C->d,
                  SPARSE_LAYOUT_ROW_MAJOR,
                  n);
  if (mkl_status != SPARSE_STATUS_SUCCESS){
      printf("mkl_sparse_spm2d error stauts:%d\n",mkl_status);
  }
}


template <typename T>
void util<T>::csr_matrix_matrix_mult(mat_csr<T> *A, mat<T> *B, mat<T> *C,T alpha,T beta)
{
  /* void mkl_scsrmm (
      const char *transa , const MKL_INT *m , const MKL_INT *n , 
      const MKL_INT *k , const float *alpha , const char *matdescra , 
      const float *val , const MKL_INT *indx , const MKL_INT *pntrb , 
      const MKL_INT *pntre , const float *b , const MKL_INT *ldb , 
      const float *beta , float *c , const MKL_INT *ldc );
  */
  const char *transa = "N";
  const char *matdescra = "GXXC";
  mklhelper<T>::mkl_csrmm(transa, &(A->nrows), &(C->ncols),
          &(A->ncols), &alpha, matdescra,
          A->values, A->cols, A->pointerB,
          A->pointerE, B->d, &(B->ncols),
          &beta, C->d, &(C->ncols));
}


template <typename T>
void util<T>::csr_matrix_delete(mat_csr<T> *M)
{
  if (M!=NULL){
    // delete M->values;
    // delete M->cols;
    // delete M->pointerB;
    // delete M->pointerE;
    mkl_free(M->values);
    mkl_free(M->cols);
    mkl_free(M->pointerB);
    //mkl_free(M->pointerE);
    delete M;
    M = NULL;
  }
}

template <typename T>
mat_csc<T> *util<T>::csc_matrix_new(MKL_INT nrows, MKL_INT ncols, long long capacity)
{
  mat_csc<T> *M = new mat_csc<T>;
  // M->pointerB = new MKL_INT[(ncols + 1)]();
  // M->pointerE = new MKL_INT[(ncols + 1)]();
  // M->rows = new MKL_INT[capacity];
  // M->values = new T[capacity];
  M->values = (T*) mkl_malloc(capacity*sizeof(T),ALIGN);
  M->rows = (MKL_INT*) mkl_malloc(capacity*sizeof(MKL_INT),ALIGN);
  M->pointerB = (MKL_INT*) mkl_malloc((ncols+1)*sizeof(MKL_INT),ALIGN);
  M->pointerE = M->pointerB + 1;
  M->nnz = capacity;
  M->nrows = nrows;
  M->ncols = ncols;
  return M;
}

template <typename T>
void util<T>::csc_matrix_delete(mat_csc<T> *M)
{
  if (M!=NULL){
    // delete M->values;
    // delete M->rows;
    // delete M->pointerB;
    // delete M->pointerE;
    mkl_free(M->values);
    mkl_free(M->rows);
    mkl_free(M->pointerB);
    //mkl_free(M->pointerE);
    delete M;
    M = NULL;
  }
}


}
