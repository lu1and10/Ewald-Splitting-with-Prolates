#ifndef GMX_MATH_PSWF_H
#define GMX_MATH_PSWF_H

#include "gromacs/utility/alignedallocator.h"

#include <iostream>
#include <unordered_map>
#include <vector>

#define Long int64_t

template<typename T>
using AlignedVector = std::vector<T, gmx::AlignedAllocator<T>>;

// blas, lapack math functions used
extern "C" {
void dgesvd_(char* jobu, char* jobvt, int* m, int* n, double* a, int* lda, double* s, double* u,
             int* ldu, double* vt, int* ldvt, double* work, int* lwork, int* info);
void dgesdd_(char* jobz, int* m, int* n, double* a, int* lda, double* s, double* u,
             int* ldu, double* vt, int* ldvt, double* work, int* lwork, int* iwork, int* info);
void dgemm_(char* TransA, char* TransB, int* M, int* N, int* K, double* alpha, double* A, int* lda,
            double* B, int* ldb, double* beta, double* C, int* ldc);
}

// legendre function
void gaussian_quadrature(int n, double* nodes, double* weights);

// chebyshev functions
void cheb_nodes_1d(int order, AlignedVector<double>& nodes, double a = 0, double b = 1);
void cheb_interp_1d(int order, AlignedVector<double>& fn_v, AlignedVector<double>& coeff);
// , double a = 0, double b = 1);

void cheb2mono(int order, int dof, AlignedVector<double>& cheb_coeff, AlignedVector<double>& mono_coeff,
               double a = 0, double b = 1);
double mono_eval(int order, const AlignedVector<double>& mono_coeff, double x);

// prolate functions
// prolate0 functor
struct Prolate0Fun;

/*
evaluate prolate0c at x, i.e., \psi_0^c(x)
*/
double prolate0_eval(double c, double x);

/*
evaluate prolate0c function integral of \int_0^r \psi_0^c(x) dx
*/
double prolate0_int_eval(double c, double r);

// approximation functions
// 1. spread window real space, tol, P, tol_coeff, domain [0.0, 1.0]
void spread_window_real_space(int P, double tol, double tol_coeff, AlignedVector<double>& coeffs);
// 2. spread window fourier space, tol, tol_coeff, domain [0.0, 1.0]
void spread_window_fourier_space(double tol, double tol_coeff, AlignedVector<double>& coeffs);
// 3. long range energy, tol, tol_coeff, domain [0.0, 1.0]
void long_range_real_energy(double tol, double tol_coeff, AlignedVector<double>& coeffs);
// 4. long range force, tol, tol_coeff, domain [0.0, 1.0]
void long_range_real_force(double tol, double tol_coeff, AlignedVector<double>& coeffs);
// 5. splitting function fourier space, tol, tol_coeff, domain [0.0, 1.0]
void splitting_function_fourier_space(double tol, double tol_coeff, AlignedVector<double>& coeffs);

#endif  // GMX_MATH_PSWF_H
