#ifndef SP_SIGN_HPP
#define SP_SIGN_HPP
#include <cstdlib>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <algorithm>
#include <ctime>
#include <cstring>
#include <omp.h>
#include "mklutil.h"

namespace sp_sign_matrix {

typedef struct
{
    uint64_t state;
    uint64_t inc;
} pcg32_random_t;

template<typename FP>
uint32_t pcg32_random_r(pcg32_random_t *rng)
{
    uint64_t oldstate = rng->state;
    // Advance internal state
    rng->state = oldstate * 6364136223846793005ULL + (rng->inc | 1);
    // Calculate output function (XSH RR), uses old state for max ILP
    uint32_t xorshifted = ((oldstate >> 18u) ^ oldstate) >> 27u;
    uint32_t rot = oldstate >> 59u;
    return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
}


template<typename FP>
MKL_INT unique(MKL_INT *a, MKL_INT n)
{
    // Assert a is sorted.
    if (n <= 1)
        return n;
    //std::sort(a,a+n,cmpINT);
    std::sort(a,a+n,[](MKL_INT x,MKL_INT y){return x < y;});
    MKL_INT ret = 0;
    for (MKL_INT i = 1; i < n; ++i)
    {
        if (*(a + ret) != *(a + i))
        {
            ++ret;
            *(a + ret) = *(a + i);
        }
    }
    return ret + 1;
}


template<typename FP>
char in_array(MKL_INT n, MKL_INT *a, MKL_INT e)
{
    for (MKL_INT i = 0; i < n; ++i)
    {
        if (a[i] == e)
            return 1;
    }
    return 0;
}

template<typename FP>
void rand_select(MKL_INT n, MKL_INT k, MKL_INT *a, pcg32_random_t *rng)//mt19937
{   //n = m,k = eta
    MKL_INT r, i;
    for (i = 0; i < k; ++i)
    {
        do
        {
            r = rand() % n; //r = [0,n-1]
        } while (in_array<FP>(i, a, r));
        a[i] = r;
    }
}


template<typename FP>
void rand_sign_mat_csc(MKL_INT n, MKL_INT eta, MKL_INT *unique_pos, mkl_util::mat_csc<FP> *spMat)
{
    for (MKL_INT i = 0; i < n; ++i)
    {
        MKL_INT row_start = i * eta;
        for (MKL_INT j = 0; j < eta; ++j)
        {
            MKL_INT idx = row_start + j;
            float val = (MKL_INT)(2 * (rand() % 2) - 1);
            spMat->values[idx] = val;
            spMat->rows[idx] = unique_pos[idx];
        }
        spMat->pointerB[i] = row_start;
        //spMat->pointerE[i] = row_start + eta;
    }
    spMat->pointerB[n] = n * eta;
    spMat->nnz = n * eta;
}


template<typename FP>
MKL_INT sp_sign_gen_csc(MKL_INT m, MKL_INT n, MKL_INT eta, MKL_INT *unique_pos, mkl_util::mat_csc<FP> *spMat)
{
    MKL_INT i;
    MKL_INT nnz = eta * n;
#pragma omp parallel for
    for (i = 0; i < n; ++i)
    {
        srand(time(0) ^ i);
        rand_select<FP>(m, eta, unique_pos + i * eta, 0);
    }
    rand_sign_mat_csc<FP>(n, eta, unique_pos, spMat);
    return unique<FP>(unique_pos, nnz);
}

}

#endif