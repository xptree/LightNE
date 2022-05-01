#pragma once
#include <iostream>
#include <cassert>
#include <set>
#include "mklfrsvds.h"

namespace mkl_frsvds{

template <typename FP>
MKLfrsvds<FP>::MKLfrsvds(mkl_util::mat_csr<FP> *A,MKL_INT rank,MKL_INT q,MKL_INT s,bool upper,bool analyze,bool sparse_project) {
  Stopwatch timer_init;
  std::cout << "init frsvds" << std::endl;
  this->n = A->nrows;
  this->rank = rank;
  this->q = q;
  this->s = s;
  this->upper = upper;
  this->analyze = analyze;
  this->sparse_project = sparse_project;
  S = mkl_util::util<FP>::matrix_new(rank,1);
  matU = mkl_util::util<FP>::matrix_new(n,rank);
  matV = NULL;
  if (upper){
    descrA.type = SPARSE_MATRIX_TYPE_SYMMETRIC;
    descrA.mode = SPARSE_FILL_MODE_UPPER;
    descrA.diag = SPARSE_DIAG_NON_UNIT;
  }
  else{
    descrA.type = SPARSE_MATRIX_TYPE_GENERAL;
  }
  
  sparse_status_t status;
  std::cout << "calling mkl_sparse_create_csr" << std::endl;
  
  status = mklhelper<FP>::mkl_sparse_create_csr(&csrA,
    SPARSE_INDEX_BASE_ZERO,
    n,
    n,
    A->pointerB,
    A->pointerE,
    A->cols,
    A->values);
  std::cout << "create sparse csr matrix done, status = " << status << ",n = " << n << std::endl;
  if (status != SPARSE_STATUS_SUCCESS){
    printf("mkl_sparse_s_create_csr error\n");
  }
  if (analyze){
    status = mkl_sparse_set_mm_hint(csrA,SPARSE_OPERATION_NON_TRANSPOSE,descrA,SPARSE_LAYOUT_ROW_MAJOR,rank,(q+1)*2);
    std::cout << "set sparse mm hint done, status=" << status << std::endl;
    status = mkl_sparse_set_memory_hint(csrA, SPARSE_MEMORY_NONE);
    std::cout << "set sparse memory hint done, status=" << status << std::endl;
  }

  status = mkl_sparse_optimize(csrA);
  if (status != SPARSE_STATUS_SUCCESS){
    printf("mkl_sparse_optimize error\n");
  }
  std::cout << "Elapsed time: " << timer_init.elapsed() << "s" << std::endl;
}

template <typename FP>
MKLfrsvds<FP>::~MKLfrsvds() {
  if (S != NULL) {
    mkl_util::util<FP>::matrix_delete(S);
  }
  if (matU != NULL) {
    mkl_util::util<FP>::matrix_delete(matU);
  }
  if (matV != NULL){
    mkl_util::util<FP>::matrix_delete(matV);
  }
}

template <typename FP>
void MKLfrsvds<FP>::run() {
  if (q == 0)
  {
    std::cout<<"Pass parameter q should be large than 0"<<std::endl;
    return;
  }
  MKL_INT l = rank + s;
  std::cout << "fast randomized symmetric svds have l:"<<l<<" q:"<<q<<std::endl;
  Stopwatch timer;
  Stopwatch timer_step;
  mkl_util::mat<FP> *Q = mkl_util::util<FP>::matrix_new(n, l);
  mkl_util::mat<FP> *Qt = mkl_util::util<FP>::matrix_new(n, l);
  mkl_util::mat<FP> *UU = mkl_util::util<FP>::matrix_new(l, l);
  mkl_util::mat<FP> *SS = mkl_util::util<FP>::matrix_new(l, 1);
  mkl_util::mat<FP> *VV = mkl_util::util<FP>::matrix_new(l, l);
  if (!sparse_project){
    std::cout<<"initialize_random_gaussian_matrix begin"<<std::endl;
    mkl_util::util<FP>::initialize_random_gaussian_matrix(Qt);
    std::cout<<"initialize_random_gaussian_matrix cost time:"<<timer_step.elapsed()<<" s"<<std::endl;
    mkl_util::util<FP>::sparse_csr_mm(csrA,Qt,Q,1.0,0,descrA);
    std::cout<<"csr mm cost time:"<<timer_step.elapsed()<<" s"<<std::endl;
  }
  else{
    MKL_INT eta = 8;
    MKL_INT *unique_pos = (MKL_INT *)calloc(eta * l,sizeof(MKL_INT));
    mkl_util::mat_csc<FP> *spMat = mkl_util::util<FP>::csc_matrix_new(n,l,eta*l);
    MKL_INT unique_num = sp_sign_matrix::sp_sign_gen_csc(n, l, eta, unique_pos, spMat); // unique_num <= eta*l
    free(unique_pos);
    std::cout<<"nnz is l*eta:"<<l*eta<<",unique num:"<<unique_num<<std::endl;
    
    sparse_matrix_t cscB,csrB;
    struct matrix_descr descrB;
    descrB.type = SPARSE_MATRIX_TYPE_GENERAL;
    sparse_status_t status;
    std::cout << "calling mkl_sparse_create_csr" << std::endl;
    status = mklhelper<FP>::mkl_sparse_create_csc(&cscB,
      SPARSE_INDEX_BASE_ZERO,
      n,
      l,
      spMat->pointerB,
      spMat->pointerB+1,
      spMat->rows,
      spMat->values);
    std::cout << "create sparse csc matrix done, status = " << status << ",n = " << n << std::endl;
    if (status != SPARSE_STATUS_SUCCESS){
      printf("mkl_sparse_s_create_csr error\n");
    }

    status = mkl_sparse_convert_csr(cscB,SPARSE_OPERATION_NON_TRANSPOSE,&csrB);
    if (status != SPARSE_STATUS_SUCCESS){
      printf("mkl_sparse_convert_csr error\n");
    }
    status = mkl_sparse_set_memory_hint(csrB, SPARSE_MEMORY_NONE);
    if (status != SPARSE_STATUS_SUCCESS){
      printf("mkl_sparse_set_memory_hint error\n");
    }
    status = mkl_sparse_optimize(csrB);
    if (status != SPARSE_STATUS_SUCCESS){
      printf("mkl_sparse_optimize error\n");
    }

    std::cout<<"initialize_sparse_projection_matrix cost time:"<<timer_step.elapsed()<<" s"<<std::endl;
    //mkl_util::util<FP>::sparse_csr_spm2d(csrA,csrB,Q,1.0,0.0,descrA,descrB);
    mkl_util::util<FP>::sparse_csr_spmmd(csrA,csrB,Q);
    std::cout<<"spmmd cost time:"<<timer_step.elapsed()<<" s"<<std::endl;
    //status = mkl_sparse_destroy(cscB);
    status = mkl_sparse_destroy(csrB);
    std::cout << "destroy sparse projection matrix done, status=" << status << std::endl;
  }
  
  mkl_util::util<FP>::eigSVD(Q, Qt, SS, VV);
  std::cout<<"eigSVD cost time:"<<timer_step.elapsed()<<" s"<<std::endl;
  MKL_INT i;
  MKL_INT niter = q;
  Stopwatch timer_iter;
  for(i=1;i<=niter;i++)
  {
    mkl_util::util<FP>::sparse_csr_mm(csrA,Qt,Q,1.0,0,descrA);
    std::cout<<"mkl_sparse_mm cost time:"<<timer_iter.elapsed()<<" s"<<std::endl;
    mkl_util::util<FP>::eigSVD(Q, Qt, SS, VV);
    std::cout<<"eigSVD cost time:"<<timer_iter.elapsed()<<" s"<<std::endl;
    //mkl_util::util<FP>::QR_factorization_getQ_inplace(Q);
  }
  std::cout<<"iteration in frsvds cost time:"<<timer_step.elapsed()<<" s"<<std::endl;
  MKL_INT inds[rank]; 
  for(i=s;i<rank+s;i++)
  {
    inds[i-s] = rank+s-(i-s)-1;
    //inds[i-s] = i-s;
  }
  mkl_util::util<FP>::matrix_copy_columns(Qt, inds, matU);
  std::cout<<"matrix copy columns:"<<timer_step.elapsed()<<" s"<<std::endl;
  
  mkl_util::util<FP>::matrix_get_selected_rows(SS, inds, S);
  mkl_util::util<FP>::matrix_delete(Qt);
  mkl_util::util<FP>::matrix_delete(UU);
  mkl_util::util<FP>::matrix_delete(SS);
  mkl_util::util<FP>::matrix_delete(VV);

  std::cout<<"time cost(in frsvds):"<<timer.elapsed()<<std::endl;
  return ;
}


} // namespace mkl_frsvds

