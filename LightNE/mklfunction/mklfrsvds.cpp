#include <iostream>
#include <cassert>
#include <set>
#include "mklfrsvds.h"

#include "pbbslib/random.h"
#include "pbbslib/sequence_ops.h"

namespace mkl_frsvds{

template <typename FP>
MKLfrsvds<FP>::MKLfrsvds(mkl_util::mat_csr<FP> *A,MKL_INT rank,MKL_INT q,MKL_INT s,bool upper,bool analyze, bool random_project_only, size_t sparse_project, float sparse_project_s) {
  Stopwatch timer_init;
  std::cout << "init frsvds" << std::endl;
  this->n = A->nrows;
  this->rank = rank;
  this->q = q;
  this->s = s;
  this->upper = upper;
  this->analyze = analyze;
  this->sparse_project = sparse_project;
  this->random_project_only = random_project_only;
  this->sparse_project_s = sparse_project_s;
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

//here apply when A is symmetric and when q = 1, the algorithm is equal to mklredsvd
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
    mkl_util::util<FP>::initialize_random_gaussian_matrix(Qt);
    std::cout<<"initialize_random_gaussian_matrix cost time:"<<timer_step.elapsed()<<" s"<<std::endl;
    mkl_util::util<FP>::sparse_csr_mm(csrA,Qt,Q,1.0,0,descrA);
    std::cout<<"csr mm cost time:"<<timer_step.elapsed()<<" s"<<std::endl;
  }
  else if(sparse_project == 1){
    // https://web.stanford.edu/~hastie/Papers/Ping/KDD06_rp.pdf
    FP s = sqrt(n);
    if (sparse_project_s >= 1.0) {
      s = static_cast<FP>(sparse_project_s);
    }
    FP density = 1. / s;
    std::cout << "very sparse random projection with s = " << s << " and density (1/s) = " << density << std::endl;
    int* non_zeros_per_row = new int[n];
    VSLStreamStatePtr stream;
    sparse_status_t mkl_status;
    lapack_int lapack_status;
    vslNewStream( &stream, VSL_BRNG_SFMT19937, 777 );
    int err = viRngBinomial(VSL_RNG_METHOD_BINOMIAL_BTPE, //method
                        stream, //stream
                        n, //n
                        non_zeros_per_row, //r
                        l, //n_trial
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

    int* tmp = new int[n * l];
    parallel_for(0, n, [&] (MKL_INT i) {
      for (size_t j=0; j < l; ++j) {
        tmp[i * l + j] = j;
      }
      // Fisher–Yates_shuffle
      auto our_seed = seed.fork(i);
      for (size_t j=0; j<non_zeros_per_row[i]; ++j) {
        int k = j + our_seed.rand() % (l - j);
        our_seed = our_seed.next();
        std::swap(tmp[i * l + j], tmp[i * l + k]);
        O_col_idx[O_rows_start[i] + j] = static_cast<MKL_INT>(tmp[i * l + j]);
        O_value[O_rows_start[i] + j] = (our_seed.rand() % 2 == 0) ? -sqrt(s) : sqrt(s);
        our_seed = our_seed.next();
      }
      // if (i == 0) {
      //   for (size_t j=0; j<non_zeros_per_row[i]; ++j) {
      //       std::cout << i << " " << O_col_idx[O_rows_start[i] + j] << " " << O_value[O_rows_start[i] + j] << std::endl;
      //   }
      // }
    }, 1024);
    delete[] non_zeros_per_row;
    delete[] tmp;
    std::cout << "sampling very sparse random matrix done, nnz= " << O_rows_start[n] << std::endl;
    sparse_matrix_t csrO;
    std::cout << "calling mkl_sparse_s_create_csr" << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    mkl_status = mklhelper<FP>::mkl_sparse_create_csr(&csrO,
        SPARSE_INDEX_BASE_ZERO,
        n,
        l,
        O_rows_start,
        O_rows_start+1,
        O_col_idx,
        O_value);
    assert(mkl_status == SPARSE_STATUS_SUCCESS);
    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> elapsed = finish - start;
    std::cout << "Elapsed time: " << elapsed.count() << " s" << std::endl;
    std::cout << "create sparse csr matrix done, status=" << mkl_status << " n= " << n << " l= " << l << " nnz= " << O_rows_start[n] << std::endl;
    
    //struct matrix_descr descrO;
    //descrO.type = SPARSE_MATRIX_TYPE_GENERAL;
    mkl_status = mkl_sparse_set_memory_hint(csrO, SPARSE_MEMORY_NONE);
    if (mkl_status != SPARSE_STATUS_SUCCESS){
      printf("mkl_sparse_set_memory_hint error\n");
    }
    mkl_status = mkl_sparse_optimize(csrO);
    if (mkl_status != SPARSE_STATUS_SUCCESS){
      printf("mkl_sparse_optimize error\n");
    }
    auto spmmd_start = std::chrono::high_resolution_clock::now();
    mkl_util::util<FP>::sparse_csr_spmmd(csrA,csrO,Q);
    assert(mkl_status == SPARSE_STATUS_SUCCESS);
    std::cout << "compute sample matrix of Q = A^T * O  = A * O done (because A^T = A)" << std::endl;
    auto spmmd_finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> spmmd_elapsed = spmmd_finish - spmmd_start;
    std::cout << "Elapsed time (spmmd): " << spmmd_elapsed.count() << " s" << std::endl;
    mkl_status = mkl_sparse_destroy(csrO);
    std::cout << "destroy sparse projection matrix done, status=" << mkl_status << std::endl;
  }
  else{
    MKL_INT eta = static_cast<MKL_INT>(sparse_project_s);
    MKL_INT *unique_pos = (MKL_INT *)calloc(eta * l,sizeof(MKL_INT));
    mkl_util::mat_csc<FP> *spMat = mkl_util::util<FP>::csc_matrix_new(n,l,eta*l);
    MKL_INT unique_num = sp_sign_matrix::sp_sign_gen_csc(n, l, eta, unique_pos, spMat); // unique_num <= eta*l
    free(unique_pos);
    std::cout<<"nnz is l*eta:"<<l*eta<<",unique num:"<<unique_num<<std::endl;
    
    sparse_matrix_t cscB,csrB;
    //struct matrix_descr descrB;
    //descrB.type = SPARSE_MATRIX_TYPE_GENERAL;
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
  if (random_project_only) {
    std::cout << "only random random projected matrix." << std::endl;
    matU = Q;
    return;
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
  mkl_util::mat<FP> *M = mkl_util::util<FP>::matrix_new(l,l);
  mkl_util::util<FP>::matrix_transpose_matrix_mult(Qt,Q,M,1.0,0);
  std::cout<<"mtmm cost time:"<<timer_step.elapsed()<<" s"<<std::endl;

  mkl_util::vec<FP> *work = mkl_util::util<FP>::vector_new(l);
  lapack_int lapack_status = mklhelper<FP>::LAPACKE_gesvd(LAPACK_ROW_MAJOR,'S', 'N', l, l, M->d, l, SS->d, UU->d, l, NULL, l, work->d);//may exist some bug
  mkl_util::util<FP>::vector_delete(work);
  //lapack_int lapack_status = mklhelper<FP>::LAPACKE_gesdd(LAPACK_ROW_MAJOR, 'S', l, l,M->d,l, SS->d, UU->d, l, VV->d, l);//gesdd slower than gesvd,gesvd get bug
  if (lapack_status!=0)
  {
    std::cout<<"some Error happen in the svd,info:"<<lapack_status<<std::endl;
    exit(1);
  }
  std::cout<<"small matrix svd cost time:"<<timer_step.elapsed()<<" s"<<std::endl;

  mkl_util::util<FP>::matrix_delete(Q);
  mkl_util::util<FP>::matrix_delete(M);
  MKL_INT inds[rank]; 
  for(i=s;i<rank+s;i++)
  {
    //inds[i-s] = rank+s-(i-s)-1;
    inds[i-s] = i-s;
  }
  mkl_util::mat<FP> *temp = mkl_util::util<FP>::matrix_new(l, rank);
  mkl_util::util<FP>::matrix_get_selected_columns(UU, inds, temp);
  mkl_util::util<FP>::matrix_matrix_mult(Qt, temp, matU,1.0,0.0);
  std::cout<<"matrix matirx mult:"<<timer_step.elapsed()<<" s"<<std::endl;
  
  mkl_util::util<FP>::matrix_get_selected_rows(SS, inds, S);
  mkl_util::util<FP>::matrix_delete(Qt);
  mkl_util::util<FP>::matrix_delete(UU);
  mkl_util::util<FP>::matrix_delete(SS);
  mkl_util::util<FP>::matrix_delete(VV);
  mkl_util::util<FP>::matrix_delete(temp);

  std::cout<<"total time cost(in frsvds):"<<timer.elapsed()<<std::endl;
  return ;
}


} // namespace mkl_frsvds

