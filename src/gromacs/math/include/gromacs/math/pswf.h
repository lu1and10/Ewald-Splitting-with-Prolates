#ifndef GMX_MATH_PSWF_H
#define GMX_MATH_PSWF_H

#include "gromacs/utility/alignedallocator.h"
#include "gromacs/utility/real.h"

#include <iostream>
#include <unordered_map>
#include <vector>

#define Long int64_t
#define MAX_CHEB_ORDER 30
#define MAX_MONO_ORDER 16

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
double mono_eval(int order, const double* mono_coeff, double x);
//double cheb_eval(int order, const AlignedVector<double>& cheb_coeff, double x, double a = 0, double b = 1);
double cheb_eval(int order, const double* cheb_coeff, double x, double a = 0, double b = 1);

// prolate functions
void prolc180_der3(double eps, double& der3);
// prolate0 functor
struct Prolate0Fun;

double prolate0_eval_derivative(double c, double x);
/*
evaluate prolate0c at x, i.e., \psi_0^c(x)
*/
double prolate0_eval(double c, double x);

/*
evaluate prolate0c function integral of \int_0^r \psi_0^c(x) dx
*/
double prolate0_int_eval(double c, double r);

// approximation functions
// 1. spread window and derivative real space, tol, P, tol_coeff, domain [0.0, 1.0]
void spread_window_real_space_der_mono(int P, double tol, double tol_coeff, AlignedVector<real>& coeffs, double& c);
void spread_window_real_space_der_cheb(int P, double tol, double tol_coeff, AlignedVector<double>& coeffs, double& c);
void spread_window_real_space_mono(int P, double tol, double tol_coeff, AlignedVector<real>& coeffs, double& c);
void spread_window_real_space_cheb(int P, double tol, double tol_coeff, AlignedVector<double>& coeffs, double& c);
void spread_window_real_space(int P, double tol, double tol_coeff, AlignedVector<double>& coeffs, double& c);
// 2. spread window fourier space, tol, tol_coeff, domain [0.0, 1.0]
void spread_window_fourier_space(double tol, double tol_coeff, AlignedVector<real>& coeffs);
// 3. long range energy, tol, tol_coeff, domain [0.0, 1.0]
void long_range_real_energy_cheb(double tol, double tol_coeff, AlignedVector<real>& coeffs, double& c, double& c0);
void long_range_real_energy(double tol, double tol_coeff, AlignedVector<double>& coeffs, double& c, double& c0);
// 4. long range force, tol, tol_coeff, domain [0.0, 1.0]
void long_range_real_force_cheb(double tol, double tol_coeff, AlignedVector<real>& coeffs);
void long_range_real_force(double tol, double tol_coeff, AlignedVector<double>& coeffs);
// 5. splitting function fourier space, tol, tol_coeff, domain [0.0, 1.0]
void splitting_function_fourier_space_cheb(double tol, double tol_coeff, AlignedVector<real>& coeffs, double& lambda);
void splitting_function_fourier_space(double tol, double tol_coeff, AlignedVector<double>& coeffs, double& lambda);
// 6. splitting function real space, tol, tol_coeff, domain [0.0, 1.0]
void splitting_function_cheb(double tol, double tol_coeff, AlignedVector<real>& coeffs, double& c, double& c0, double& psi0);
void splitting_function(double tol, double tol_coeff, AlignedVector<double>& coeffs, double& c, double& c0, double& psi0);

double spread_window_ref(double c, int P, int i, double x);
double spread_window_der_ref(double c, int P, int i, double x);
double spread_window_fourier_ref(double c, double x);
double splitting_function_fourier_space_ref(double c, double c0, double lambda, double x);
double splitting_function_real_space_ref(double c, double c0, double x);
#endif  // GMX_MATH_PSWF_H