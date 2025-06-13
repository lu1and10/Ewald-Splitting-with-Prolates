
#include "gmxpre.h"

#include "gromacs/math/pswf.h"
#include "gromacs/simd/simd.h"
//#include "gromacs/simd/simd_math.h"
#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <unordered_map>
#include <utility>
#include <vector>

//using namespace std;

// start of legendre functions
static inline void legepol(double x, int n, double& pol, double& der) {
    double pkm1 = 1.0;
    double pk = x;
    double pkp1;

    if (n == 0) {
        pol = 1.0;
        der = 0.0;
        return;
    }

    if (n == 1) {
        pol = x;
        der = 1.0;
        return;
    }

    pk = 1.0;
    pkp1 = x;

    for (int k = 1; k < n; ++k) {
        pkm1 = pk;
        pk = pkp1;
        pkp1 = ((2 * k + 1) * x * pk - k * pkm1) / (k + 1);
    }

    pol = pkp1;
    der = n * (x * pkp1 - pk) / (x * x - 1);
}

static inline void legetayl(double pol, double der, double x, double h, int n, int k, double& sum,
                            double& sumder) {
    double done = 1.0;
    double q0 = pol;
    double q1 = der * h;
    double q2 = (2 * x * der - n * (n + done) * pol) / (1 - x * x);
    q2 = q2 * h * h / 2;

    sum = q0 + q1 + q2;
    sumder = q1 / h + q2 * 2 / h;

    if (k <= 2) return;

    double qi = q1;
    double qip1 = q2;

    for (int i = 1; i <= k - 2; ++i) {
        double d = 2 * x * (i + 1) * (i + 1) / h * qip1 - (n * (n + done) - i * (i + 1)) * qi;
        d = d / (i + 1) / (i + 2) * h * h / (1 - x * x);
        double qip2 = d;

        sum += qip2;
        sumder += d * (i + 2) / h;

        qi = qip1;
        qip1 = qip2;
    }
}

/* Constructs Gaussian quadrature of order n.
   itype=1 => both roots (ts) and weights (whts) are computed.
   itype=0 => only roots (ts) are computed. */
static inline void legerts(int itype, int n, double* ts, double* whts) {
    int k = 30;
    double d = 1.0;
    double d2 = d + 1.0e-24;
    if (d2 != d) {
        k = 54;
    }

    int half = n / 2;
    int ifodd = n - 2 * half;
    double pi_val = atan(1.0) * 4.0;
    double h = pi_val / (2.0 * n);

    /* Initial approximations (for i >= n/2+1) */
    int ii = 0;
    for (int i = 1; i <= n; i++) {
        if (i < (n / 2 + 1)) {
            continue;
        }
        ii++;
        double t = (2.0 * i - 1.0) * h;
        ts[ii - 1] = -cos(t);
    }

    /* Start from center: find roots one by one via Newton updates */
    double pol = 1.0, der = 0.0;
    double x0 = 0.0;
    legepol(x0, n, pol, der);
    double x1 = ts[0];

    int n2 = (n + 1) / 2;
    double pol3 = pol, der3 = der;

    for (int kk = 1; kk <= n2; kk++) {
        if ((ifodd == 1) && (kk == 1)) {
            ts[kk - 1] = x0;
            if (itype > 0) {
                whts[kk - 1] = der;
            }
            x0 = x1;
            x1 = ts[kk];
            pol3 = pol;
            der3 = der;
            continue;
        }

        /* Newton iteration */
        int ifstop = 0;
        for (int i = 1; i <= 10; i++) {
            double hh = x1 - x0;

            legetayl(pol3, der3, x0, hh, n, k, pol, der);
            x1 = x1 - pol / der;

            if (fabs(pol) < 1.0e-12) {
                ifstop++;
            }
            if (ifstop == 3) {
                break;
            }
        }

        ts[kk - 1] = x1;
        if (itype > 0) {
            whts[kk - 1] = der;
        }

        x0 = x1;
        x1 = ts[kk];
        pol3 = pol;
        der3 = der;
    }

    /* Mirror roots around 0: fill second half of ts[] */
    for (int i = n2; i >= 1; i--) {
        ts[i - 1 + half] = ts[i - 1];
    }
    for (int i = 1; i <= half; i++) {
        ts[i - 1] = -ts[n - i];
    }
    if (itype <= 0) {
        return;
    }

    /* Mirror weights similarly */
    for (int i = n2; i >= 1; i--) {
        whts[i - 1 + half] = whts[i - 1];
    }
    for (int i = 1; i <= half; i++) {
        whts[i - 1] = whts[n - i];
    }

    /* Compute final weights = 2 / (1 - ts[i]^2) / (der[i]^2) */
    for (int i = 0; i < n; i++) {
        double tmp = 1.0 - ts[i] * ts[i];
        whts[i] = 2.0 / tmp / (whts[i] * whts[i]);
    }
}

static inline void legepols(double x, int n, double* pols) {
    double pkm1 = 1.0;
    double pk = x;

    if (n == 0) {
        pols[0] = 1.0;
        return;
    }

    if (n == 1) {
        pols[0] = 1.0;
        pols[1] = x;
        return;
    }

    pols[0] = 1.0;
    pols[1] = x;

    for (int k = 1; k < n; ++k) {
        double pkp1 = ((2 * k + 1) * x * pk - k * pkm1) / (k + 1);
        pols[k + 1] = pkp1;
        pkm1 = pk;
        pk = pkp1;
    }
}

// TODO: legepols() is not tested yet.
// only itype !=2 is tested.
static inline void legeexps(int itype, int n, double* x, AlignedVector<AlignedVector<double>>& u,
                            AlignedVector<AlignedVector<double>>& v, double* whts) {
    int itype_rts = (itype > 0) ? 1 : 0;

    // Call legerts to construct the nodes and weights of the n-point Gaussian quadrature
    legerts(itype_rts, n, x, whts);
    // gaussian_quadrature(n, x, whts);

    // legerts() is buggy now, below is not tested yet.
    // legerts_(&itype_rts, &n, x, whts);

    // If itype is not 2, return early
    if (itype != 2) return;

    // Construct the matrix of values of the Legendre polynomials at these nodes
    for (int i = 0; i < n; ++i) {
        AlignedVector<double> pols(n);
        legepols(x[i], n - 1, pols.data());
        for (int j = 0; j < n; ++j) {
            u[j][i] = pols[j];
        }
    }

    // Transpose u to get v
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            v[i][j] = u[j][i];
        }
    }

    // Construct the inverse u, converting the values of a function at Gaussian nodes into the
    // coefficients of a Legendre expansion of that function
    for (int i = 0; i < n; ++i) {
        double d = 1.0 * (2 * (i + 1) - 1) / 2;
        for (int j = 0; j < n; ++j) {
            u[i][j] = v[j][i] * whts[j] * d;
        }
    }
}

static inline void legeexev(double x, double& val, const double* pexp, int n) {
    double pjm2 = 1.0;
    double pjm1 = x;

    val = pexp[0] * pjm2 + pexp[1] * pjm1;

    for (int j = 2; j <= n; ++j) {
        double pj = ((2 * j - 1) * x * pjm1 - (j - 1) * pjm2) / j;
        val += pexp[j] * pj;
        pjm2 = pjm1;
        pjm1 = pj;
    }
}

static inline void legeFDER(double x, double& val, double& der, const double* pexp, int n) {
    double pjm2 = 1.0;
    double pjm1 = x;
    double derjm2 = 0.0;
    double derjm1 = 1.0;

    val = pexp[0] * pjm2 + pexp[1] * pjm1;
    der = pexp[1];

    for (int j = 2; j <= n; ++j) {
        double pj = ((2 * j - 1) * x * pjm1 - (j - 1) * pjm2) / j;
        val += pexp[j] * pj;

        double derj = (2 * j - 1) * (pjm1 + x * derjm1) - (j - 1) * derjm2;
        derj /= j;
        der += pexp[j] * derj;

        pjm2 = pjm1;
        pjm1 = pj;
        derjm2 = derjm1;
        derjm1 = derj;
    }
}

// Function to compute the Legendre polynomial p_n(x) and its derivative p'_n(x)
static inline void legendre(int n, double x, double& pn, double& pn_prime) {
    if (n == 0) {
        pn = 1.0;
        pn_prime = 0.0;
        return;
    }

    if (n == 1) {
        pn = x;
        pn_prime = 1.0;
        return;
    }

    double pn_minus1 = 1.0;  // P_0(x)
    double pn_minus2 = 0.0;  // P_-1(x)
    pn = x;                  // P_1(x)

    for (int k = 2; k <= n; ++k) {
        pn_minus2 = pn_minus1;
        pn_minus1 = pn;
        pn = ((2.0 * k - 1.0) * x * pn_minus1 - (k - 1.0) * pn_minus2) / k;
    }

    pn_prime = n * (x * pn - pn_minus1) / (x * x - 1.0);
}

// Function to compute the nodes and weights of the n-point Gaussian quadrature
void gaussian_quadrature(int n, double* nodes, double* weights) {
    constexpr double my_pi = 3.1415926535897932384626433832795028841L;
    static AlignedVector<AlignedVector<double>> precomp_nodes(1000);
    static AlignedVector<AlignedVector<double>> precomp_weights(1000);

    {  // Precompute
        assert(n < precomp_nodes.size() && n < precomp_weights.size());
        if (precomp_nodes[n].size() == 0 && precomp_weights[n].size() == 0) {
#pragma omp critical(GAUSS_QUAD)
            if (precomp_nodes[n].size() == 0 && precomp_weights[n].size() == 0) {
                AlignedVector<double> nodes1(n);
                AlignedVector<double> weights1(n);

                if (true) {  // usr original code
                    legerts(1, n, nodes1.data(), weights1.data());
                } else {  // self implementation test code
                    const double tolerance = 1e-16;
                    const int max_iterations = 100;
                    for (int i = 0; i < (n + 1) / 2; ++i) {
                        double x = std::cos(my_pi * (i + 0.75) / (n + 0.5));
                        double pn, pn_prime;

                        for (int iter = 0; iter < max_iterations; ++iter) {
                            legendre(n, x, pn, pn_prime);
                            double delta_x = -pn / pn_prime;
                            x += delta_x;

                            if (std::abs(delta_x) < tolerance) {
                                break;
                            }
                        }

                        nodes1[i] = x;
                        nodes1[n - 1 - i] = -x;
                        weights1[i] = 2.0 / ((1.0 - x * x) * pn_prime * pn_prime);
                        weights1[n - 1 - i] = weights1[i];
                    }
                }

                precomp_nodes[n].swap(nodes1);
                precomp_weights[n].swap(weights1);
            }
        }
    }

    memcpy(nodes, precomp_nodes[n].data(), n * sizeof(double));
    memcpy(weights, precomp_weights[n].data(), n * sizeof(double));
}
// end of legendre functions

// start of chebyshev functions
// Custom hash function for std::pair
struct pair_hash {
    template <class T1, class T2>
    std::size_t operator()(const std::pair<T1, T2>& p) const {
        auto hash1 = std::hash<T1>{}(p.first);
        auto hash2 = std::hash<T2>{}(p.second);
        return hash1 ^ (hash2 << 1);  // Combine the two hash values
    }
};

// Custom equality function for std::pair
struct pair_equal {
    template <class T1, class T2>
    bool operator()(const std::pair<T1, T2>& p1, const std::pair<T1, T2>& p2) const {
        return p1.first == p2.first && p1.second == p2.second;
    }
};

/**
 * Computes the gemm operation C = A * B.
 * A is NxK, B is KxM, C is NxM.
 * Using BLAS interface.
 */
static inline void gemm(int M, int N, int K, double* A, double* B, double* C) {
    char trans = 'N';
    double alpha = 1.0;
    double beta = 0.0;
    int lda = M;
    int ldb = K;
    int ldc = M;
    dgemm_(&trans, &trans, &M, &N, &K, &alpha, B, &lda, A, &ldb, &beta, C, &ldc);
}

/**
 * Computes the pseudo inverse of matrix M(n1xn2) (in row major form)
 * and returns the output M_(n2xn1). Original contents of M are destroyed.
 */
static inline void pseudo_inv(double* M, int n1, int n2, double eps, double* M_) {
    // int alignment = 1024;
    if (n1 * n2 == 0) return;
    int m = n2;
    int n = n1;
    int k = (m < n ? m : n);

    // double* tU = (double*)aligned_alloc(alignment, m * k * sizeof(double));
    // double* tS = (double*)aligned_alloc(alignment, k * sizeof(double));
    // double* tVT = (double*)aligned_alloc(alignment, k * n * sizeof(double));
    //double* tU = (double*)malloc(m * k * sizeof(double));
    // TODO: debug, when linking external lapack(mkl, openblas, etc) tU can be m*k
    // when linking internal lapack, tU needs to be m*max(m,n) otherwise it will cause
    // segmentation fault.
    // Not sure why, need to debug and investigate.
    double* tU = (double*)malloc(m * std::max<int>(m,n) * sizeof(double));
    double* tS = (double*)malloc(k * sizeof(double));
    double* tVT = (double*)malloc(k * n * sizeof(double));
    //int* iwork = (int*)malloc(8 * k * sizeof(int));
    int* iwork = (int*)malloc(8 * std::max<int>(m,n) * sizeof(int));

    // SVD
    int INFO = 0;
    //char JOBU = 'S';
    //char JOBVT = 'S';
    char JOBZ = 'S';

    int wssize = -1;
    double wkopt{0};
    //dgesvd_(&JOBU, &JOBVT, &m, &n, M, &m, tS, tU, &m, tVT, &k, &wkopt, &wssize, &INFO);
    dgesdd_(&JOBZ, &m, &n, M, &m, tS, tU, &m, tVT, &k, &wkopt, &wssize, iwork, &INFO);
    wssize = (int)wkopt;

    // double* wsbuf = (double*)aligned_alloc(alignment, wssize);
    double* wsbuf = (double*)malloc(wssize*sizeof(double));
    //dgesvd_(&JOBU, &JOBVT, &m, &n, M, &m, tS, tU, &m, tVT, &k, wsbuf, &wssize, &INFO);
    dgesdd_(&JOBZ, &m, &n, M, &m, tS, tU, &m, tVT, &k, wsbuf, &wssize, iwork, &INFO);
    free(wsbuf);

    if (INFO != 0) std::cout << "svd failed with info: " << INFO << '\n';
    assert(INFO == 0);

    double max_S = std::abs(tS[0]);
    /*
    double max_S = 0;
    for (Long i = 0; i < k; i++) {
        max_S = std::max<double>(max_S, std::abs(tS[i]));
    }
    */

    double eps_ = max_S * eps;
    for (int i = 0; i < k; i++)
        if (tS[i] < eps_)
            tS[i] = 0;
        else
            tS[i] = 1 / tS[i];

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < k; j++) {
            tU[i + j * m] *= tS[j];
        }
    }

    double alpha = 1.0, beta = 0.0;
    char trans = 'T';
    dgemm_(&trans, &trans, &n, &m, &k, &alpha, tVT, &k, tU, &m, &beta, M_, &n);
    free(tU);
    free(tS);
    free(tVT);
    free(iwork);
}
// end of math utils

// start of monomial utils
static inline void monomial_nodes_1d(int nnodes, AlignedVector<double>& nodes, double a = 0, double b = 1) {
    if (nodes.size() != nnodes) nodes.resize(nnodes);
    for (int i = 0; i < nnodes; i++) {
        nodes[i] = (double)i / (double)(nnodes - 1) * (b - a) + a;
    }
}

static inline void monomial_basis_1d(int order, const AlignedVector<double>& x, AlignedVector<double>& y,
                                     double a = 0, double b = 1) {
    int n = x.size();
    if (y.size() != order * n) y.resize(order * n);

    for (Long i = 0; i < n; i++) {
        for (int j = 0; j < order; j++) {
            y[j * n + i] = pow((x[i] - a) / (b - a), order - j - 1);
        }
    }
}

static inline void monomial_interp_1d(int order, int nnodes, AlignedVector<double>& fn_v, AlignedVector<double>& coeff,
                                      double a = 0, double b = 1) {
    /*
    static AlignedVector<AlignedVector<double>> precomp(1000);
    {  // Precompute
        assert(order < precomp.size());
        if (precomp[order].size() == 0) {
#pragma omp critical(MONOMIAL_BASIS_APPROX)
            if (precomp[order].size() == 0) {
                AlignedVector<double> x, p;
                monomial_nodes_1d(order, x, a, b);
                monomial_basis_1d(order, x, p);
                AlignedVector<double> Mp1(order * order);
                pseudo_inv(p.data(), order, order, std::numeric_limits<double>::epsilon(),
                           Mp1.data());
                precomp[order].swap(Mp1);
            }
        }
    }
    AlignedVector<double>& Mp = precomp[order];
    Long dof = fn_v.size() / order;
    */
    AlignedVector<double> x, p, Mp(order * nnodes);
    //monomial_nodes_1d(nnodes, x, a, b);
    cheb_nodes_1d(nnodes, x, a, b);
    monomial_basis_1d(order, x, p);
    // TODO: check solve in lsq sense, i.e., mul order matters (us)(v^t x)
    pseudo_inv(p.data(), order, nnodes, std::numeric_limits<double>::epsilon(), Mp.data());
    Long dof  = fn_v.size() / nnodes;
    assert(fn_v.size() == dof * nnodes);
    if (coeff.size() != dof * order) coeff.resize(dof * order);
    gemm(order, dof, nnodes, fn_v.data(), Mp.data(), coeff.data());
}

// end of monomial utils
// start of actual cheb utils
void cheb_nodes_1d(int order, AlignedVector<double>& nodes, double a, double b) {
    constexpr double my_pi = 3.1415926535897932384626433832795028841L;
    if (nodes.size() != order) nodes.resize(order);
    for (int i = 0; i < order; i++) {
        nodes[i] = -cos((i + (double)0.5) * my_pi / order) * (double)0.5 + (double)0.5;
        nodes[i] = nodes[i] * (b - a) + a;
    }
}

static inline void cheb_basis_1d(int order, const AlignedVector<double>& x, AlignedVector<double>& y,
                                 double a = 0, double b = 1) {
    int n = x.size();
    if (y.size() != order * n) y.resize(order * n);

    if (order > 0) {
        for (Long i = 0; i < n; i++) {
            y[i] = 1.0;
        }
    }
    if (order > 1) {
        for (Long i = 0; i < n; i++) {
            y[i + n] = x[i] * 2 / (b - a) - 2 * a / (b - a) - 1;
        }
    }
    for (int i = 2; i < order; i++) {
        for (Long j = 0; j < n; j++) {
            y[i * n + j] = 2 * y[n + j] * y[i * n - 1 * n + j] - y[i * n - 2 * n + j];
        }
    }
}

//void cheb_interp_1d(int order, AlignedVector<double>& fn_v, AlignedVector<double>& coeff, double a, double b) {
void cheb_interp_1d(int order, AlignedVector<double>& fn_v, AlignedVector<double>& coeff) {
    // static unordered_map<pair<double, double>, AlignedVector<AlignedVector<double>>, pair_hash, pair_equal>
    // precomp_map; pair<double, double> key = make_pair(a, b); if (precomp_map.find(key) ==
    // precomp_map.end()) {
    //     precomp_map.emplace(key, AlignedVector<AlignedVector<double>>(1000));
    // }
    // AlignedVector<AlignedVector<double>> &precomp = precomp_map[key];
    static AlignedVector<AlignedVector<double>> precomp(1000);
    {  // Precompute
        assert(order < precomp.size());
        if (precomp[order].size() == 0) {
#pragma omp critical(CHEB_BASIS_APPROX)
            if (precomp[order].size() == 0) {
                AlignedVector<double> x, p;
                cheb_nodes_1d(order, x);
                cheb_basis_1d(order, x, p);
                AlignedVector<double> Mp1(order * order);
                pseudo_inv(p.data(), order, order, std::numeric_limits<double>::epsilon(),
                           Mp1.data());
                precomp[order].swap(Mp1);
            }
        }
    }
    AlignedVector<double>& Mp = precomp[order];

    Long dof = fn_v.size() / order;
    assert(fn_v.size() == dof * order);
    if (coeff.size() != dof * order) coeff.resize(dof * order);

    gemm(order, dof, order, fn_v.data(), Mp.data(), coeff.data());
}

static inline void generate_polynomial_coeffs(int n, AlignedVector<double>& coeffs, double a = 0,
                                              double b = 1) {
    // Resize the matrix to n x n
    if (coeffs.size() != n * n)
        coeffs.resize(n * n, 0.0);
    else
        std::fill(coeffs.begin(), coeffs.end(), 0.0);

    // Initialize the first two polynomials
    coeffs[0] = 1.0;
    coeffs[1] = 0.0;
    if (n > 1) {
        coeffs[n + 1] = 2.0 / (b - a);           // y_2 = 2/(b-a) x
        coeffs[n + 0] = -2.0 * a / (b - a) - 1;  // y_2 = -2a/(b-a) - 1
    }

    // Generate the rest of the polynomials using the recursive relation
    for (int i = 2; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (j > 0) {
                coeffs[i * n + j] += 2.0 * (2.0 / (b - a)) *
                                     coeffs[(i - 1) * n + j - 1];  // 2 * ((2.0/(b-a))x) * y_{i-1}
            }
            coeffs[i * n + j] -= coeffs[(i - 2) * n + j];  // - y_{i-2}
        }
        for (int j = 0; j < n; ++j) {
            coeffs[i * n + j] -= 2.0 * (2.0 * a / (b - a) + 1) *
                                 coeffs[(i - 1) * n + j];  // 2 * (-2a/(b-a) - 1) * y_{i-1}
        }
    }
}

void cheb2mono(int order, int dof, AlignedVector<double>& cheb_coeff, AlignedVector<double>& mono_coeff, double a,
               double b) {
    assert(cheb_coeff.size() == order * dof);
    if (mono_coeff.size() != order * dof) mono_coeff.resize(order * dof);

    static std::unordered_map<std::pair<double, double>, AlignedVector<AlignedVector<double>>, pair_hash, pair_equal>
        precomp_map;

    std::pair<double, double> key{a, b};

    if (precomp_map.find(key) == precomp_map.end()) {
#pragma omp critical(CHEB_TO_MONO)
        if (precomp_map.find(key) == precomp_map.end()) {
            precomp_map.emplace(key, AlignedVector<AlignedVector<double>>(1000));
        }
    }
    AlignedVector<AlignedVector<double>>& precomp = precomp_map[key];
    {  // Precompute
        assert(order < precomp.size());
        if (precomp[order].size() == 0) {
#pragma omp critical(CHEB_TO_MONO)
            if (precomp[order].size() == 0) {
                AlignedVector<double> Mp1(order * order, 0.0);
                generate_polynomial_coeffs(order, Mp1, a, b);
                precomp[order].swap(Mp1);
            }
        }
    }
    AlignedVector<double>& Mp = precomp[order];

    gemm(order, dof, order, cheb_coeff.data(), Mp.data(), mono_coeff.data());
}

template <int order>
double mono_eval_template(const double* mono_coeff, double x) {
    if (order == 0) return 0.0;
    if (order == 1) return mono_coeff[0];

    double y = mono_coeff[0];
    for (int i = 1; i < order; i++) {
        y = mono_coeff[i] + x * y;
    }

    return y;
}

template <int poly_order>
double mono_eval_dispatch(int order, const double* mono_coeff, double x) {
    static_assert(poly_order <= MAX_CHEB_ORDER, "poly_order must be in the range [0, 30]");
    if constexpr (poly_order == 0) {
        return mono_eval_template<0>(mono_coeff, x);
    }
    else {
        if(poly_order == order){
            return mono_eval_template<poly_order>(mono_coeff, x);
        }
        else {
            return mono_eval_dispatch<poly_order-1>(order, mono_coeff, x);
        }

    }
}

double mono_eval(int order, const double* mono_coeff, double x) {
    return mono_eval_dispatch<MAX_CHEB_ORDER>(order, mono_coeff, x);
}

// clenshaw cheb eval
template <int order>
double cheb_eval_template(const double* cheb_coeff, double x, double a, double b) {
    if (order == 0) return 0.0;
    if (order == 1) return cheb_coeff[0];

    // Transform x from [a,b] to [-1,1]
    double y = 2.0 * (x - a) / (b - a) - 1.0;
    double y2 = 2.0 * y;

    double b0 = 0.0;
    double b1 = 0.0;
    double b2 = 0.0;

    for (int i = order - 1; i > 0; --i) {
        b2 = b1;
        b1 = b0;
        b0 = cheb_coeff[i] + y2 * b1 - b2;
    }

    return cheb_coeff[0] + b0 * y - b1;
}

template <int poly_order>
double cheb_eval_dispatch(int order, const double* cheb_coeff, double x, double a, double b) {
    static_assert(poly_order <= MAX_CHEB_ORDER, "poly_order must be in the range [0, 30]");
    if constexpr (poly_order == 0) {
        return cheb_eval_template<0>(cheb_coeff, x, a, b);
    }
    else {
        if(poly_order == order){
            return cheb_eval_template<poly_order>(cheb_coeff, x, a, b);
        }
        else {
            return cheb_eval_dispatch<poly_order-1>(order, cheb_coeff, x, a, b);
        }

    }
}

//double cheb_eval(int order, const AlignedVector<double>& cheb_coeff, double x, double a, double b) {
double cheb_eval(int order, const double* cheb_coeff, double x, double a, double b) {
   return cheb_eval_dispatch<MAX_CHEB_ORDER>(order, cheb_coeff, x, a, b);
}
// end of chebyshev functions

// start of prolate functions
void prolc180_der3(double eps, double& der3) {

    static const std::array<double, 180> der3s = {
        0.0                  , 3.703231117106314e-02, 2.662796075011415e-01, 5.941926542747007e-01, 9.578230754832373e-01, 1.363569812949696e+00,
        1.807582627232425e+00, 2.287040787443315e+00, 2.800489841625932e+00, 3.346520825211228e+00, 3.925129833524817e+00, 4.535419879018613e+00,
        5.177538088593701e+00, 5.851057619952580e+00, 6.555804054828174e+00, 7.292020430656394e+00, 8.059158990368712e+00, 8.857123665436557e+00,
        9.686327541088655e+00, 1.054640955930842e+01, 1.143707141760266e+01, 1.235843748588681e+01, 1.331074380051023e+01, 1.429316645410095e+01,
        1.530644212280254e+01, 1.634980229052360e+01, 1.742371527350574e+01, 1.852786959134436e+01, 1.966241028647265e+01, 2.082706565691709e+01,
        2.202156399063638e+01, 2.324513764895337e+01, 2.450154874132168e+01, 2.578557879197864e+01, 2.710219152702352e+01, 2.844590643457592e+01,
        2.982194872189250e+01, 3.122457703819981e+01, 3.265927724478793e+01, 3.412004831578323e+01, 3.561263588011575e+01, 3.713704155571862e+01,
        3.869326681306944e+01, 4.027479309954505e+01, 4.188123512956217e+01, 4.352576155609020e+01, 4.519494867085516e+01, 4.689544558088133e+01,
        4.862725312364265e+01, 5.038308226775531e+01, 5.216996738943379e+01, 5.398790913473069e+01, 5.583690809846664e+01, 5.771696483563476e+01,
        5.962015154378219e+01, 6.155414190868980e+01, 6.351893636867634e+01, 6.551453532837742e+01, 6.753250221415664e+01, 6.958958399152446e+01,
        7.166878031319368e+01, 7.377852843589304e+01, 7.591882864408177e+01, 7.808968120601168e+01, 8.028188878922057e+01, 8.251372002218120e+01,
        8.476665370485364e+01, 8.704988763041084e+01, 8.936342198942501e+01, 9.170725696750367e+01, 9.408139273542218e+01, 9.647574810864796e+01,
        9.891035968318585e+01, 1.013649391732796e+02, 1.038495680706917e+02, 1.063748312950304e+02, 1.089196851143706e+02, 1.114945887021435e+02,
        1.140885801820887e+02, 1.167234579055465e+02, 1.193883857149036e+02, 1.220720250413416e+02, 1.247969275635160e+02, 1.275402908406401e+02,
        1.303251684172720e+02, 1.331282559724378e+02, 1.359611431651965e+02, 1.388238300708471e+02, 1.417163167603507e+02, 1.446386032954563e+02,
        1.475782236729200e+02, 1.505599848644160e+02, 1.535715460806204e+02, 1.566000662021011e+02, 1.596711024010027e+02, 1.627588477107742e+02,
        1.658893590967149e+02, 1.690363297860595e+02, 1.722128508870634e+02, 1.754189224418921e+02, 1.786545444888072e+02, 1.819197170667437e+02,
        1.852144402114015e+02, 1.885387139581953e+02, 1.918925383387225e+02, 1.952759133924808e+02, 1.986743765316800e+02, 2.021167283277167e+02,
        2.055886308747634e+02, 2.090752480704547e+02, 2.126061275354775e+02, 2.161514728445827e+02, 2.197413293189910e+02, 2.233454028450144e+02,
        2.269787784335390e+02, 2.306414561065548e+02, 2.343491422119853e+02, 2.380705482795840e+02, 2.418212564954385e+02, 2.456012668766033e+02,
        2.494105794440804e+02, 2.532491942146669e+02, 2.571171112093926e+02, 2.610143304358681e+02, 2.649241522962268e+02, 2.688798518867251e+02,
        2.728648537691562e+02, 2.768791579444873e+02, 2.809055686954829e+02, 2.849783533576689e+02, 2.890804403554123e+02, 2.931942620075851e+02,
        2.973548295757224e+02, 3.015268840118008e+02, 3.057279930461074e+02, 3.099762194632530e+02, 3.142355613580597e+02, 3.185422684529161e+02,
        3.228598432462443e+02, 3.272064726902580e+02, 3.315821567930542e+02, 3.360057012097200e+02, 3.404396183022963e+02, 3.449025900686188e+02,
        3.493946165380561e+02, 3.539156977033438e+02, 3.584658335865259e+02, 3.630450242038053e+02, 3.676532695407853e+02, 3.722905695962574e+02,
        3.769569244097179e+02, 3.816523339874616e+02, 3.863767983079612e+02, 3.911100281057601e+02, 3.958924783306446e+02, 4.007039833307977e+02,
        4.055445431251419e+02, 4.103933743959938e+02, 4.152919201395830e+02, 4.202195206860208e+02, 4.251550223528558e+02, 4.301406088745175e+02,
        4.351552502298094e+02, 4.401774223148202e+02, 4.452500497108158e+02, 4.503299610661473e+02, 4.554605744971888e+02, 4.605982251667677e+02,
        4.657646839606519e+02, 4.709822146642819e+02, 4.762064127721131e+02, 4.814819295722471e+02, 4.867638670342317e+02, 4.920746126195583e+02,
        4.974370467592532e+02, 5.028055317088541e+02, 5.082028248088587e+02, 5.136289260479356e+02, 5.191072088380275e+02, 5.245910494770286e+02,};

        if (eps < 1.0e-18) eps = 1e-18;
        double d = -log10(eps);
        int i = static_cast<int>(d * 10 + 0.1);
        der3 = der3s[i - 1];
}

static inline void prolc180(double eps, double& c) {
    static const std::array<double, 180> cs = {
        0.43368E-16, 0.10048E+01, 0.17298E+01, 0.22271E+01, 0.26382E+01, 0.30035E+01, 0.33409E+01,
        0.36598E+01, 0.39658E+01, 0.42621E+01, 0.45513E+01, 0.48347E+01, 0.51136E+01, 0.53887E+01,
        0.56606E+01, 0.59299E+01, 0.61968E+01, 0.64616E+01, 0.67247E+01, 0.69862E+01, 0.72462E+01,
        0.75049E+01, 0.77625E+01, 0.80189E+01, 0.82744E+01, 0.85289E+01, 0.87826E+01, 0.90355E+01,
        0.92877E+01, 0.95392E+01, 0.97900E+01, 0.10040E+02, 0.10290E+02, 0.10539E+02, 0.10788E+02,
        0.11036E+02, 0.11284E+02, 0.11531E+02, 0.11778E+02, 0.12024E+02, 0.12270E+02, 0.12516E+02,
        0.12762E+02, 0.13007E+02, 0.13251E+02, 0.13496E+02, 0.13740E+02, 0.13984E+02, 0.14228E+02,
        0.14471E+02, 0.14714E+02, 0.14957E+02, 0.15200E+02, 0.15443E+02, 0.15685E+02, 0.15927E+02,
        0.16169E+02, 0.16411E+02, 0.16652E+02, 0.16894E+02, 0.17135E+02, 0.17376E+02, 0.17617E+02,
        0.17858E+02, 0.18098E+02, 0.18339E+02, 0.18579E+02, 0.18819E+02, 0.19059E+02, 0.19299E+02,
        0.19539E+02, 0.19778E+02, 0.20018E+02, 0.20257E+02, 0.20496E+02, 0.20736E+02, 0.20975E+02,
        0.21214E+02, 0.21452E+02, 0.21691E+02, 0.21930E+02, 0.22168E+02, 0.22407E+02, 0.22645E+02,
        0.22884E+02, 0.23122E+02, 0.23360E+02, 0.23598E+02, 0.23836E+02, 0.24074E+02, 0.24311E+02,
        0.24549E+02, 0.24787E+02, 0.25024E+02, 0.25262E+02, 0.25499E+02, 0.25737E+02, 0.25974E+02,
        0.26211E+02, 0.26448E+02, 0.26685E+02, 0.26922E+02, 0.27159E+02, 0.27396E+02, 0.27633E+02,
        0.27870E+02, 0.28106E+02, 0.28343E+02, 0.28580E+02, 0.28816E+02, 0.29053E+02, 0.29289E+02,
        0.29526E+02, 0.29762E+02, 0.29998E+02, 0.30234E+02, 0.30471E+02, 0.30707E+02, 0.30943E+02,
        0.31179E+02, 0.31415E+02, 0.31651E+02, 0.31887E+02, 0.32123E+02, 0.32358E+02, 0.32594E+02,
        0.32830E+02, 0.33066E+02, 0.33301E+02, 0.33537E+02, 0.33773E+02, 0.34008E+02, 0.34244E+02,
        0.34479E+02, 0.34714E+02, 0.34950E+02, 0.35185E+02, 0.35421E+02, 0.35656E+02, 0.35891E+02,
        0.36126E+02, 0.36362E+02, 0.36597E+02, 0.36832E+02, 0.37067E+02, 0.37302E+02, 0.37537E+02,
        0.37772E+02, 0.38007E+02, 0.38242E+02, 0.38477E+02, 0.38712E+02, 0.38947E+02, 0.39181E+02,
        0.39416E+02, 0.39651E+02, 0.39886E+02, 0.40120E+02, 0.40355E+02, 0.40590E+02, 0.40824E+02,
        0.41059E+02, 0.41294E+02, 0.41528E+02, 0.41763E+02, 0.41997E+02, 0.42232E+02, 0.42466E+02,
        0.42700E+02, 0.42935E+02, 0.43169E+02, 0.43404E+02, 0.43638E+02, 0.43872E+02, 0.44107E+02,
        0.44341E+02, 0.44575E+02, 0.44809E+02, 0.45044E+02, 0.45278E+02};

    if (eps < 1.0e-18) eps = 1e-18;
    double d = -log10(eps);
    int i = static_cast<int>(d * 10 + 0.1);
    c = cs[i - 1];
}

static inline void prosinin(double c, const double* ts, const double* whts, const double* fs,
                            double x, int n, double& rint, double& derrint) {
    rint = 0.0;
    derrint = 0.0;

    for (int i = 0; i < n; ++i) {
        double diff = x - ts[i];
        double sin_term = sin(c * diff);
        double cos_term = cos(c * diff);

        rint += whts[i] * fs[i] * sin_term / diff;

        derrint += whts[i] * fs[i] / (diff * diff) * (c * diff * cos_term - sin_term);
    }
}

static inline void prolcoef(double rlam, int k, double c, double& alpha0, double& beta0,
                            double& gamma0, double& alpha, double& beta, double& gamma) {
    double d = k * (k - 1);
    d = d / (2 * k + 1) / (2 * k - 1);
    double uk = d;

    d = (k + 1) * (k + 1);
    d = d / (2 * k + 3);
    double d2 = k * k;
    d2 = d2 / (2 * k - 1);
    double vk = (d + d2) / (2 * k + 1);

    d = (k + 1) * (k + 2);
    d = d / (2 * k + 1) / (2 * k + 3);
    double wk = d;

    alpha = -c * c * uk;
    beta = rlam - k * (k + 1) - c * c * vk;
    gamma = -c * c * wk;

    alpha0 = uk;
    beta0 = vk;
    gamma0 = wk;
}

static inline void prolmatr(double* as, double* bs, double* cs, int n, double c, double rlam,
                            int ifsymm, int ifodd) {
    double done = 1.0;
    double half = done / 2.0;
    int k = 0;

    if (ifodd > 0) {
        for (int k0 = 1; k0 <= n + 2; k0 += 2) {
            k++;
            double alpha0, beta0, gamma0, alpha, beta, gamma;
            prolcoef(rlam, k0, c, alpha0, beta0, gamma0, alpha, beta, gamma);

            as[k - 1] = alpha;
            bs[k - 1] = beta;
            cs[k - 1] = gamma;

            if (ifsymm != 0) {
                if (k0 > 1) {
                    as[k - 1] = as[k - 1] / std::sqrt(k0 - 2 + half) * std::sqrt(k0 + half);
                }
                cs[k - 1] = cs[k - 1] * std::sqrt(k0 + half) / std::sqrt(k0 + half + 2);
            }
        }
    } else {
        for (int k0 = 0; k0 <= n + 2; k0 += 2) {
            k++;
            double alpha0, beta0, gamma0, alpha, beta, gamma;
            prolcoef(rlam, k0, c, alpha0, beta0, gamma0, alpha, beta, gamma);

            as[k - 1] = alpha;
            bs[k - 1] = beta;
            cs[k - 1] = gamma;

            if (ifsymm != 0) {
                if (k0 != 0) {
                    as[k - 1] = as[k - 1] / std::sqrt(k0 - 2 + half) * std::sqrt(k0 + half);
                }
                cs[k - 1] = cs[k - 1] * std::sqrt(k0 + half) / std::sqrt(k0 + half + 2);
            }
        }
    }
}

static inline void prolql1(int n, double* d, double* e, int& ierr) {
    ierr = 0;
    if (n == 1) return;

    for (int i = 1; i < n; ++i) {
        e[i - 1] = e[i];
    }
    e[n - 1] = 0.0;

    for (int l = 0; l < n; ++l) {
        int j = 0;
        while (true) {
            int m;
            for (m = l; m < n - 1; ++m) {
                double tst1 = std::abs(d[m]) + std::abs(d[m + 1]);
                double tst2 = tst1 + std::abs(e[m]);
                if (tst2 == tst1) break;
            }

            if (m == l) break;
            if (j == 30) {
                ierr = l + 1;
                return;
            }
            ++j;

            double g = (d[l + 1] - d[l]) / (2.0 * e[l]);
            double r = std::sqrt(g * g + 1.0);
            g = d[m] - d[l] + e[l] / (g + std::copysign(r, g));
            double s = 1.0;
            double c = 1.0;
            double p = 0.0;

            for (int i = m - 1; i >= l; --i) {
                double f = s * e[i];
                double b = c * e[i];
                r = std::sqrt(f * f + g * g);
                e[i + 1] = r;
                if (r == 0.0) {
                    d[i + 1] -= p;
                    e[m] = 0.0;
                    break;
                }
                s = f / r;
                c = g / r;
                g = d[i + 1] - p;
                r = (d[i] - g) * s + 2.0 * c * b;
                p = s * r;
                d[i + 1] = g + p;
                g = c * r - b;
            }

            if (r == 0.0) break;
            d[l] -= p;
            e[l] = g;
            e[m] = 0.0;
        }

        if (l == 0) continue;
        for (int i = l; i > 0; --i) {
            if (d[i] >= d[i - 1]) break;
            std::swap(d[i], d[i - 1]);
        }
    }
}

static inline void prolfact(double* a, double* b, double* c, int n, double* u, double* v,
                            double* w) {
    // Eliminate down
    for (int i = 0; i < n - 1; ++i) {
        double d = c[i + 1] / a[i];
        a[i + 1] -= b[i] * d;
        u[i] = d;
    }

    // Eliminate up
    for (int i = n - 1; i > 0; --i) {
        double d = b[i - 1] / a[i];
        v[i] = d;
    }

    // Scale the diagonal
    double done = 1.0;
    for (int i = 0; i < n; ++i) {
        w[i] = done / a[i];
    }
}

static inline void prolsolv(const double* u, const double* v, const double* w, int n, double* rhs) {
    // Eliminate down
    for (int i = 0; i < n - 1; ++i) {
        rhs[i + 1] -= u[i] * rhs[i];
    }

    // Eliminate up
    for (int i = n - 1; i > 0; --i) {
        rhs[i - 1] -= rhs[i] * v[i];
    }

    // Scale
    for (int i = 0; i < n; ++i) {
        rhs[i] *= w[i];
    }
}

static inline void prolfun0(int& ier, int n, double c, double* as, double* bs, double* cs,
                            double* xk, double* u, double* v, double* w, double eps, int& nterms,
                            double& rkhi) {
    ier = 0;
    double delta = 1.0e-8;
    int ifsymm = 1;
    int numit = 4;
    double rlam = 0;
    int ifodd = -1;

    prolmatr(as, bs, cs, n, c, rlam, ifsymm, ifodd);

    prolql1(n / 2, bs, as, ier);
    if (ier != 0) {
        ier = 2048;
        return;
    }

    rkhi = -bs[n / 2 - 1];
    rlam = -bs[n / 2 - 1] + delta;

    std::fill(xk, xk + n, 1.0);

    prolmatr(as, bs, cs, n, c, rlam, ifsymm, ifodd);

    prolfact(bs, cs, as, n / 2, u, v, w);

    for (int ijk = 0; ijk < numit; ++ijk) {
        prolsolv(u, v, w, n / 2, xk);

        double d = 0;
        for (int j = 0; j < n / 2; ++j) {
            d += xk[j] * xk[j];
        }

        d = std::sqrt(d);
        for (int j = 0; j < n / 2; ++j) {
            xk[j] /= d;
        }

        double err = 0;
        for (int j = 0; j < n / 2; ++j) {
            err += (as[j] - xk[j]) * (as[j] - xk[j]);
            as[j] = xk[j];
        }
        err = std::sqrt(err);
    }

    double half = 0.5;
    for (int i = 0; i < n / 2; ++i) {
        if (std::abs(xk[i]) > eps) nterms = i + 1;
        xk[i] *= std::sqrt(i * 2 + half);
        cs[i] = xk[i];
    }

    int j = 0;
    for (int i = 0; i <= nterms; ++i) {
        xk[j++] = cs[i];
        xk[j++] = 0;
    }

    nterms *= 2;
}

static inline void prolps0i(int& ier, double c, double* w, int lenw, int& nterms, int& ltot,
                            double& rkhi) {
    static const std::array<int, 20> ns = {48,  64,  80,  92,  106, 120, 130, 144, 156, 168,
                                           178, 190, 202, 214, 224, 236, 248, 258, 268, 280};

    double eps = 1.0e-16;
    int n = static_cast<int>(c * 3);
    n = n / 2;

    int i = static_cast<int>(c / 10);
    if (i <= 19) n = ns[i];

    ier = 0;
    int ixk = 1;
    int lxk = n + 2;

    int ias = ixk + lxk;
    int las = n + 2;

    int ibs = ias + las;
    int lbs = n + 2;

    int ics = ibs + lbs;
    int lcs = n + 2;

    int iu = ics + lcs;
    int lu = n + 2;

    int iv = iu + lu;
    int lv = n + 2;

    int iw = iv + lv;
    int lw = n + 2;

    ltot = iw + lw;

    if (ltot >= lenw) {
        ier = 512;
        return;
    }

    // Call to prolfun0 (to be implemented)
    prolfun0(ier, n, c, w + ias - 1, w + ibs - 1, w + ics - 1, w + ixk - 1, w + iu - 1, w + iv - 1,
             w + iw - 1, eps, nterms, rkhi);

    if (ier != 0) return;
}

static inline void prol0ini(int& ier, double c, double* w, double& rlam20, double& rkhi, int lenw,
                            int& keep, int& ltot) {
    ier = 0;
    double thresh = 45;
    int iw = 11;
    w[0] = iw + 0.1;
    w[8] = thresh;

    // Create the data to be used in the evaluation of the function ψ^c_0(x) for x ∈ [-1,1]
    int nterms = 0;
    prolps0i(ier, c, w + iw - 1, lenw, nterms, ltot, rkhi);

    if (ier != 0) return;

    // If c > thresh, do not prepare data for the evaluation of ψ^c_0 outside the interval [-1,1]
    if (c >= thresh) {
        w[7] = c;
        w[4] = nterms + 0.1;
        keep = nterms + 3;
        return;
    }

    // Create the data to be used in the evaluation of the function ψ^c_0(x) for x outside the
    // interval [-1,1]
    int ngauss = nterms * 2;
    int lw = nterms + 2;
    int its = iw + lw;
    int lts = ngauss + 2;
    int iwhts = its + lts;
    int lwhts = ngauss + 2;
    int ifs = iwhts + lwhts;
    int lfs = ngauss + 2;

    keep = ifs + lfs;
    if (keep > ltot) ltot = keep;
    if (keep >= lenw) {
        ier = 1024;
        return;
    }

    w[1] = its + 0.1;
    w[2] = iwhts + 0.1;
    w[3] = ifs + 0.1;

    int itype = 1;
    AlignedVector<AlignedVector<double>> u;
    AlignedVector<AlignedVector<double>> v;
    legeexps(itype, ngauss, w + its - 1, u, v, w + iwhts - 1);

    // Evaluate the prolate function at the Gaussian nodes
    for (int i = 0; i < ngauss; ++i) {
        legeexev(w[its + i - 2], w[ifs + i - 2], w + iw - 1, nterms - 1);
    }

    // Calculate the eigenvalue corresponding to ψ^c_0
    double rlam = 0;
    double x0 = 0;
    double f0;
    legeexev(x0, f0, w + iw - 1, nterms - 1);
    double der;
    prosinin(c, w + its - 1, w + iwhts - 1, w + ifs - 1, x0, ngauss, rlam, der);

    rlam = rlam / f0;
    rlam20 = rlam;

    w[4] = nterms + 0.1;
    w[5] = ngauss + 0.1;
    w[6] = rlam;
    w[7] = c;
}

static inline void prol0eva(double x, const double* w, double& psi0, double& derpsi0) {
    int iw = static_cast<int>(w[0]);
    int its = static_cast<int>(w[1]);
    int iwhts = static_cast<int>(w[2]);
    int ifs = static_cast<int>(w[3]);

    int nterms = static_cast<int>(w[4]);
    int ngauss = static_cast<int>(w[5]);
    double rlam = w[6];
    double c = w[7];
    double thresh = w[8];

    if (std::abs(x) > 1) {
        if (c < thresh - 1.0e-10) {
            psi0 = 0;
            derpsi0 = 0;
            return;
        }

        prosinin(c, &w[its - 1], &w[iwhts - 1], &w[ifs - 1], x, ngauss, psi0, derpsi0);
        psi0 /= rlam;
        derpsi0 /= rlam;
        return;
    }

    legeFDER(x, psi0, derpsi0, &w[iw - 1], nterms - 2);
    // to match chebfun psi0, needs a factor of sqrt(2)
    // psi0 = sqrt(2.0) * psi0;
    // derpsi0 = sqrt(2.0) * derpsi0;
}

static inline void prol0int0r(const double* w, double r, double& val) {
    static int npts = 200;
    static int itype = 1;
    double derpsi0;
    static AlignedVector<double> xs(npts, 0), ws(npts, 0), fvals(npts, 0);
    static int need_init = 1;
    AlignedVector<AlignedVector<double>> u;
    AlignedVector<AlignedVector<double>> v;

    // since xs, ws, fval of size 200 are static
    // only need to get nodes and weights once
    if (need_init) {
#pragma omp critical(PROL0INT0R)
        if (need_init) {
            legeexps(itype, npts, xs.data(), u, v, ws.data());
            need_init = 0;
        }
    }

    // Scale the nodes and weights to [0, r]
    double xs_r;
    for (int i = 0; i < npts; ++i) {
        xs_r = (xs[i] + 1) * r / 2;
        prol0eva(xs_r, w, fvals[i], derpsi0);
    }

    val = 0;
    for (int i = 0; i < npts; ++i) {
        val += ws[i] * r / 2 * fvals[i];
    }
}

struct Prolate0Fun {
    Prolate0Fun() = default;

    inline Prolate0Fun(double c_, int lenw_) : c(c_), lenw(lenw_) {
        int ier;
        workarray.resize(lenw);
        prol0ini(ier, c, workarray.data(), rlam20, rkhi, lenw, keep, ltot);
        // prol0ini_(&ier, &c, workarray.data(), &rlam20, &rkhi, &lenw, &keep, &ltot);
        if (ier) throw std::runtime_error("Unable to init Prolate0Fun");
    }

    // evaluate prolate0 function val and derivative
    inline std::pair<double, double> eval_val_derivative(double x) const {
        double psi0, derpsi0;
        prol0eva(x, workarray.data(), psi0, derpsi0);
        // prol0eva_(&x, workarray.data(), &psi0, &derpsi0);
        // std::pair<double, double> psi0_derpsi0{psi0, derpsi0};
        return {psi0, derpsi0};
    }

    // evaluate prolate0 function value
    inline double eval_val(double x) const {
        auto [val, dum] = eval_val_derivative(x);
        return val;
    }

    // evaluate prolate0 function derivative
    inline double eval_derivative(double x) const {
        auto [dum, der] = eval_val_derivative(x);
        return der;
    }

    // int_0^r prolate0(x) dx
    inline double int_eval(double r) const {
        double val;
        prol0int0r(workarray.data(), r, val);
        // prol0int0r_(workarray.data(), &r, &val);
        return val;
    }

    double c;
    int lenw, keep, ltot;
    AlignedVector<double> workarray;
    double rlam20, rkhi;
};

/*
evaluate prolate0c derivative at x, i.e., \psi_0^c(x)
*/
double prolate0_eval_derivative(double c, double x) {
    static std::unordered_map<double, Prolate0Fun> prolate0_funcs_cache;
    if (prolate0_funcs_cache.find(c) == prolate0_funcs_cache.end()) {
#pragma omp critical(PROLATE0_EVAL)
        if (prolate0_funcs_cache.find(c) == prolate0_funcs_cache.end()) {
            #ifdef MYDEBUGPRINT
            std::cout << "Creating new eval Prolate0Fun derivative for c = " << c << std::endl;
            #endif
            prolate0_funcs_cache.emplace(c, Prolate0Fun(c, 10000));
        }
    }
    return prolate0_funcs_cache[c].eval_derivative(x);
}

/*
evaluate prolate0c at x, i.e., \psi_0^c(x)
*/
double prolate0_eval(double c, double x) {
    static std::unordered_map<double, Prolate0Fun> prolate0_funcs_cache;
    if (prolate0_funcs_cache.find(c) == prolate0_funcs_cache.end()) {
#pragma omp critical(PROLATE0_EVAL)
        if (prolate0_funcs_cache.find(c) == prolate0_funcs_cache.end()) {
            #ifdef MYDEBUGPRINT
            std::cout << "Creating new eval Prolate0Fun for c = " << c << std::endl;
            #endif
            prolate0_funcs_cache.emplace(c, Prolate0Fun(c, 10000));
        }
    }
    return prolate0_funcs_cache[c].eval_val(x);
}

/*
evaluate prolate0c function integral of \int_0^r \psi_0^c(x) dx
*/
double prolate0_int_eval(double c, double r) {
    static std::unordered_map<double, Prolate0Fun> prolate0_funcs_cache;
    if (prolate0_funcs_cache.find(c) == prolate0_funcs_cache.end()) {
#pragma omp critical(PROLATE0_INT_EVAL)
        if (prolate0_funcs_cache.find(c) == prolate0_funcs_cache.end()) {
            #ifdef MYDEBUGPRINT
            std::cout << "Creating new int_eval Prolate0Fun for c = " << c << std::endl;
            #endif
            prolate0_funcs_cache.emplace(c, Prolate0Fun(c, 10000));
        }
    }
    return prolate0_funcs_cache[c].int_eval(r);
}
// end of prolate functions

// start of approximation functions
/* Returns the smallest number >= \p that is a multiple of \p factor, \p factor must be a power of 2 */
template<unsigned int factor>
static size_t roundUpToMultipleOfFactor(size_t number)
{
    static_assert(gmx::isPowerOfTwo(factor));

    /* We need to add a most factor-1 and because factor is a power of 2,
     * we get the result by masking out the bits corresponding to factor-1.
     */
    return (number + factor - 1) & ~(factor - 1);
}

void spread_window_real_space_der_mono(int P, double tol, double tol_coeff, AlignedVector<real>& coeffs, double& c) {
    prolc180(tol, c);

    int order = MAX_CHEB_ORDER;
    AlignedVector<double> nodes;
    cheb_nodes_1d(order, nodes);

    auto f = [](int P, int i, double c, double x) {
        double arg = x - P / 2.0 + i;
        arg /= P / 2.0;
        double val = prolate0_eval_derivative(c, arg)*2.0/P;
        return val;
    };
    int dof = P;
    AlignedVector<double> fn_v(dof * order);
    for (int idof = 0; idof < dof; idof++) {
        for (int i = 0; i < order; i++) {
            fn_v[idof * order + i] = f(P, idof, c, nodes[i]);
        }
    }

    AlignedVector<double> cheb_coeff;
    cheb_interp_1d(order, fn_v, cheb_coeff);
    int max_order = -1;
    // filter chebyshev coefficients
    double max_coeffs = 0.0;
    for (int idof = 0; idof < dof; idof++) {
        max_coeffs = 0.0;
        for (int i = 0; i < order; i++) {
            max_coeffs = std::max<double>(max_coeffs, std::abs(cheb_coeff[idof * order + i]));
        }
        for (int i = 0; i < order; i++) {
            if (std::abs(cheb_coeff[idof * order + i]) > tol_coeff * max_coeffs) {
                max_order = std::max<int>(max_order, i + 1);
            }
        }
    }

    #ifdef MYDEBUGPRINT
    std::cout << "spread der max_order estimated from cheb = " << max_order << std::endl;
    #endif
    if(max_order > MAX_MONO_ORDER) {
        std::cout << "max_order = " << max_order << " > MAX_MONO_ORDER = " << MAX_MONO_ORDER << std::endl;
        std::cout << "max_order set to MAX_MONO_ORDER = " << MAX_MONO_ORDER << std::endl;
        max_order = MAX_MONO_ORDER;
    }

    // now actually construct the mono coeffs with order max_order
    constexpr int c_simdWidth = GMX_SIMD_REAL_WIDTH;
    size_t padded_dof = roundUpToMultipleOfFactor<c_simdWidth>(dof);
    coeffs.resize(padded_dof * max_order, 0.0);
    //if (coeffs.size() != dof * max_order) coeffs.resize(dof * max_order);

    AlignedVector<double> coeffs_tmp(dof * max_order);
    int nnodes = (int)max_order*1.75;
    #ifdef MYDEBUGPRINT
    std::cout << "nnodes = " << nnodes << std::endl;
    #endif
    //monomial_nodes_1d(nnodes, nodes, 0, 1);
    cheb_nodes_1d(nnodes, nodes, 0, 1);
    fn_v.resize(dof * nnodes);
    for (int idof = 0; idof < dof; idof++) {
        for (int i = 0; i < nnodes; i++) {
            fn_v[idof * nnodes + i] = f(P, dof - idof - 1, c, nodes[i]);
        }
    }
    #ifdef MYDEBUGPRINT
    std::cout << "spread der dof = " << dof << " max_order = " << max_order << " nnodes = " << nnodes << std::endl;
    #endif

    //monomial_interp_1d(max_order, nnodes, fn_v, coeffs);
    monomial_interp_1d(max_order, nnodes, fn_v, coeffs_tmp);

    // copy coeffs_tmp to coeffs for simd eval
    for (int i = 0; i < dof; i++) {
        int iw = i/c_simdWidth;
        int iw_rem = i%c_simdWidth;
        int off_set = iw*c_simdWidth*max_order;
        for (int j = 0; j < max_order; j++) {
            coeffs[off_set+j*c_simdWidth+iw_rem] = coeffs_tmp[i * max_order + j];
        }
    }
}

void spread_window_real_space_der_cheb(int P, double tol, double tol_coeff, AlignedVector<double>& coeffs, double& c) {
    prolc180(tol, c);

    int order = MAX_CHEB_ORDER;
    AlignedVector<double> nodes;
    cheb_nodes_1d(order, nodes);

    auto f = [](int P, int i, double c, double x) {
        double arg = x - P / 2.0 + i;
        arg /= P / 2.0;
        double val = prolate0_eval_derivative(c, arg)*2.0/P;
        return val;
    };
    int dof = P;
    AlignedVector<double> fn_v(dof * order);
    for (int idof = 0; idof < dof; idof++) {
        for (int i = 0; i < order; i++) {
            fn_v[idof * order + i] = f(P, idof, c, nodes[i]);
        }
    }

    AlignedVector<double> cheb_coeff;
    cheb_interp_1d(order, fn_v, cheb_coeff);
    int max_order = -1;
    // filter chebyshev coefficients
    double max_coeffs = 0.0;
    for (int idof = 0; idof < dof; idof++) {
        max_coeffs = 0.0;
        for (int i = 0; i < order; i++) {
            max_coeffs = std::max<double>(max_coeffs, std::abs(cheb_coeff[idof * order + i]));
        }
        for (int i = 0; i < order; i++) {
            if (std::abs(cheb_coeff[idof * order + i]) > tol_coeff * max_coeffs) {
                max_order = std::max<int>(max_order, i + 1);
            }
        }
    }

    AlignedVector<double>& cheb_coeff_filtered = coeffs;
    if (cheb_coeff_filtered.size() != dof * max_order) cheb_coeff_filtered.resize(dof * max_order);
    for (int i = 0; i < dof; i++) {
        for (int j = 0; j < max_order; j++) {
            cheb_coeff_filtered[i * max_order + j] = cheb_coeff[i * order + j];
        }
    }
}

void spread_window_real_space_mono(int P, double tol, double tol_coeff, AlignedVector<real>& coeffs, double& c) {
    prolc180(tol, c);

    int order = MAX_CHEB_ORDER;
    AlignedVector<double> nodes;
    cheb_nodes_1d(order, nodes);

    auto f = [](int P, int i, double c, double x) {
        double arg = x - P / 2.0 + i;
        arg /= P / 2.0;
        double val = prolate0_eval(c, arg);
        return val;
    };
    int dof = P;
    AlignedVector<double> fn_v(dof * order);
    for (int idof = 0; idof < dof; idof++) {
        for (int i = 0; i < order; i++) {
            fn_v[idof * order + i] = f(P, idof, c, nodes[i]);
        }
    }

    AlignedVector<double> cheb_coeff;
    cheb_interp_1d(order, fn_v, cheb_coeff);
    int max_order = -1;
    // filter chebyshev coefficients
    double max_coeffs = 0.0;
    for (int idof = 0; idof < dof; idof++) {
        max_coeffs = 0.0;
        for (int i = 0; i < order; i++) {
            max_coeffs = std::max<double>(max_coeffs, std::abs(cheb_coeff[idof * order + i]));
        }
        for (int i = 0; i < order; i++) {
            if (std::abs(cheb_coeff[idof * order + i]) > tol_coeff * max_coeffs) {
                max_order = std::max<int>(max_order, i + 1);
            }
        }
    }

    #ifdef MYDEBUGPRINT
    std::cout << "spread max_order estimated from cheb = " << max_order << std::endl;
    #endif
    if(max_order > MAX_MONO_ORDER) {
        std::cout << "max_order = " << max_order << " > MAX_MONO_ORDER = " << MAX_MONO_ORDER << std::endl;
        std::cout << "max_order set to MAX_MONO_ORDER = " << MAX_MONO_ORDER << std::endl;
        max_order = MAX_MONO_ORDER;
    }

    // now actually construct the mono coeffs with order max_order
    constexpr int c_simdWidth = GMX_SIMD_REAL_WIDTH;
    size_t padded_dof = roundUpToMultipleOfFactor<c_simdWidth>(dof);
    #ifdef MYDEBUGPRINT
    std::cout << "padded_dof = " << padded_dof << std::endl;
    #endif
    coeffs.resize(padded_dof * max_order, 0.0);
    //if (coeffs.size() != dof * max_order) coeffs.resize(dof * max_order);

    AlignedVector<double> coeffs_tmp(dof * max_order);
    int nnodes = (int)max_order*1.75;
    #ifdef MYDEBUGPRINT
    std::cout << "nnodes = " << nnodes << std::endl;
    #endif
    //monomial_nodes_1d(nnodes, nodes, 0, 1);
    cheb_nodes_1d(nnodes, nodes, 0, 1);
    fn_v.resize(dof * nnodes);
    for (int idof = 0; idof < dof; idof++) {
        for (int i = 0; i < nnodes; i++) {
            fn_v[idof * nnodes + i] = f(P, dof - idof - 1, c, nodes[i]);
        }
    }

    #ifdef MYDEBUGPRINT
    std::cout << "spread dof = " << dof << " max_order = " << max_order << " nnodes = " << nnodes << std::endl;
    #endif

    //monomial_interp_1d(max_order, nnodes, fn_v, coeffs);
    monomial_interp_1d(max_order, nnodes, fn_v, coeffs_tmp);

    // copy coeffs_tmp to coeffs for simd eval
    for (int i = 0; i < dof; i++) {
        int iw = i/c_simdWidth;
        int iw_rem = i%c_simdWidth;
        int off_set = iw*c_simdWidth*max_order;
        for (int j = 0; j < max_order; j++) {
            coeffs[off_set+j*c_simdWidth+iw_rem] = coeffs_tmp[i * max_order + j];
        }
    }
}

void spread_window_real_space_cheb(int P, double tol, double tol_coeff, AlignedVector<double>& coeffs, double& c) {
    prolc180(tol, c);

    int order = MAX_CHEB_ORDER;
    AlignedVector<double> nodes;
    cheb_nodes_1d(order, nodes);

    auto f = [](int P, int i, double c, double x) {
        double arg = x - P / 2.0 + i;
        arg /= P / 2.0;
        double val = prolate0_eval(c, arg);
        return val;
    };
    int dof = P;
    AlignedVector<double> fn_v(dof * order);
    for (int idof = 0; idof < dof; idof++) {
        for (int i = 0; i < order; i++) {
            fn_v[idof * order + i] = f(P, idof, c, nodes[i]);
        }
    }

    AlignedVector<double> cheb_coeff;
    cheb_interp_1d(order, fn_v, cheb_coeff);
    int max_order = -1;
    // filter chebyshev coefficients
    double max_coeffs = 0.0;
    for (int idof = 0; idof < dof; idof++) {
        max_coeffs = 0.0;
        for (int i = 0; i < order; i++) {
            max_coeffs = std::max<double>(max_coeffs, std::abs(cheb_coeff[idof * order + i]));
        }
        for (int i = 0; i < order; i++) {
            if (std::abs(cheb_coeff[idof * order + i]) > tol_coeff * max_coeffs) {
                max_order = std::max<int>(max_order, i + 1);
            }
        }
    }

    AlignedVector<double>& cheb_coeff_filtered = coeffs;
    if (cheb_coeff_filtered.size() != dof * max_order) cheb_coeff_filtered.resize(dof * max_order);
    for (int i = 0; i < dof; i++) {
        for (int j = 0; j < max_order; j++) {
            cheb_coeff_filtered[i * max_order + j] = cheb_coeff[i * order + j];
        }
    }
}

void spread_window_real_space(int P, double tol, double tol_coeff, AlignedVector<double>& coeffs, double& c) {
    prolc180(tol, c);

    int order = MAX_CHEB_ORDER;
    AlignedVector<double> nodes;
    cheb_nodes_1d(order, nodes);

    auto f = [](int P, int i, double c, double x) {
        double arg = x - P / 2.0 + i;
        arg /= P / 2.0;
        double val = prolate0_eval(c, arg);
        return val;
    };
    int dof = P;
    AlignedVector<double> fn_v(dof * order);
    for (int idof = 0; idof < dof; idof++) {
        for (int i = 0; i < order; i++) {
            fn_v[idof * order + i] = f(P, idof, c, nodes[i]);
        }
    }

    AlignedVector<double> cheb_coeff;
    cheb_interp_1d(order, fn_v, cheb_coeff);
    int max_order = -1;
    // filter chebyshev coefficients
    AlignedVector<double> max_coeffs(dof, 0.0);
    for (int idof = 0; idof < dof; idof++) {
        for (int i = 0; i < order; i++) {
            max_coeffs[idof] =
                std::max<double>(max_coeffs[idof], std::abs(cheb_coeff[idof * order + i]));
        }
        for (int i = 0; i < order; i++) {
            if (std::abs(cheb_coeff[idof * order + i]) < tol_coeff * max_coeffs[idof]) {
                cheb_coeff[idof * order + i] = 0;
            } else {
                max_order = std::max<int>(max_order, i + 1);
            }
        }
    }

    AlignedVector<double> mono_coeff;
    cheb2mono(order, dof, cheb_coeff, mono_coeff);

    AlignedVector<double>& mono_coeff_filtered = coeffs;
    if (mono_coeff_filtered.size() != dof * max_order) mono_coeff_filtered.resize(dof * max_order);
    for (int i = 0; i < dof; i++) {
        for (int j = 0; j < max_order; j++) {
            mono_coeff_filtered[i * max_order + j] = mono_coeff[i * order + j];
        }
    }
}

void spread_window_fourier_space(double tol, double tol_coeff, AlignedVector<real>& coeffs) {
    double c;
    prolc180(tol, c);

    int quad_npts = 200;
    AlignedVector<double> xs(quad_npts, 0), ws(quad_npts, 0);
    gaussian_quadrature(quad_npts, xs.data(), ws.data());
    double lambda = 0.0;
    for (int i = 0; i < quad_npts; i++) {
        lambda += ws[i] * prolate0_eval(c, xs[i]) * std::cos(c * xs[i] * 0.5);
    }
    lambda /= prolate0_eval(c, 0.5);

    #ifdef MYDEBUGPRINT
    std::cout << "spread window fourier lambda = " << lambda << std::endl;
    #endif

    int order = MAX_CHEB_ORDER;
    AlignedVector<double> nodes;
    cheb_nodes_1d(order, nodes);

    auto f = [](double lambda, double c, double x) {
        double val = lambda * prolate0_eval(c, x);
        return val;
    };
    int dof = 1;
    AlignedVector<double> fn_v(dof * order);
    for (int idof = 0; idof < dof; idof++) {
        for (int i = 0; i < order; i++) {
            fn_v[idof * order + i] = f(lambda, c, nodes[i]);
        }
    }

    AlignedVector<double> cheb_coeff;
    cheb_interp_1d(order, fn_v, cheb_coeff);
    int max_order = -1;
    // filter chebyshev coefficients
    AlignedVector<double> max_coeffs(dof, 0.0);
    for (int idof = 0; idof < dof; idof++) {
        for (int i = 0; i < order; i++) {
            max_coeffs[idof] =
                std::max<double>(max_coeffs[idof], std::abs(cheb_coeff[idof * order + i]));
        }
        for (int i = 0; i < order; i++) {
            if (std::abs(cheb_coeff[idof * order + i]) < tol_coeff * max_coeffs[idof]) {
                cheb_coeff[idof * order + i] = 0;
            } else {
                max_order = std::max<int>(max_order, i + 1);
            }
        }
    }

    AlignedVector<double> mono_coeff;
    cheb2mono(order, dof, cheb_coeff, mono_coeff);

    AlignedVector<real>& mono_coeff_filtered = coeffs;
    if (mono_coeff_filtered.size() != dof * max_order) mono_coeff_filtered.resize(dof * max_order);
    for (int i = 0; i < dof; i++) {
        for (int j = 0; j < max_order; j++) {
            mono_coeff_filtered[i * max_order + j] = mono_coeff[i * order + j];
        }
    }
}

void long_range_real_energy_cheb(double tol, double tol_coeff, AlignedVector<real>& coeffs, double& c, double& c0) {
    prolc180(tol, c);

    c0 = prolate0_int_eval(c, 1.0);

    int order = MAX_CHEB_ORDER;
    AlignedVector<double> nodes;
    cheb_nodes_1d(order, nodes);

    auto f = [](double c0, double c, double x) {
        double val = prolate0_int_eval(c, x) / c0;
        return val;
    };
    int dof = 1;
    AlignedVector<double> fn_v(dof * order);
    for (int idof = 0; idof < dof; idof++) {
        for (int i = 0; i < order; i++) {
            fn_v[idof * order + i] = f(c0, c, nodes[i]);
        }
    }

    AlignedVector<double> cheb_coeff;
    cheb_interp_1d(order, fn_v, cheb_coeff);
    int max_order = -1;
    // filter chebyshev coefficients
    double max_coeffs = 0.0;
    for (int idof = 0; idof < dof; idof++) {
        max_coeffs = 0.0;
        for (int i = 0; i < order; i++) {
            max_coeffs = std::max<double>(max_coeffs, std::abs(cheb_coeff[idof * order + i]));
        }
        for (int i = 0; i < order; i++) {
            if (std::abs(cheb_coeff[idof * order + i]) > tol_coeff * max_coeffs) {
                max_order = std::max<int>(max_order, i + 1);
            }
        }
    }

    AlignedVector<real>& cheb_coeff_filtered = coeffs;
    if (cheb_coeff_filtered.size() != dof * max_order) cheb_coeff_filtered.resize(dof * max_order);
    for (int i = 0; i < dof; i++) {
        for (int j = 0; j < max_order; j++) {
            cheb_coeff_filtered[i * max_order + j] = cheb_coeff[i * order + j];
        }
    }
}

void long_range_real_energy_mono(double tol, double tol_coeff, AlignedVector<real>& coeffs, double& c, double& c0) {
    prolc180(tol, c);

    c0 = prolate0_int_eval(c, 1.0);

    int order = MAX_CHEB_ORDER;
    AlignedVector<double> nodes;
    cheb_nodes_1d(order, nodes);

    auto f = [](double c0, double c, double x) {
        double val = prolate0_int_eval(c, x) / c0;
        return val;
    };
    int dof = 1;
    AlignedVector<double> fn_v(dof * order);
    for (int idof = 0; idof < dof; idof++) {
        for (int i = 0; i < order; i++) {
            fn_v[idof * order + i] = f(c0, c, nodes[i]);
        }
    }

    AlignedVector<double> cheb_coeff;
    cheb_interp_1d(order, fn_v, cheb_coeff);
    int max_order = -1;
    // filter chebyshev coefficients
    double max_coeffs = 0.0;
    for (int idof = 0; idof < dof; idof++) {
        max_coeffs = 0.0;
        for (int i = 0; i < order; i++) {
            max_coeffs = std::max<double>(max_coeffs, std::abs(cheb_coeff[idof * order + i]));
        }
        for (int i = 0; i < order; i++) {
            if (std::abs(cheb_coeff[idof * order + i]) > tol_coeff * max_coeffs) {
                max_order = std::max<int>(max_order, i + 1);
            }
        }
    }

    if(max_order > MAX_MONO_ORDER) {
        std::cout << "max_order = " << max_order << " > MAX_MONO_ORDER = " << MAX_MONO_ORDER << std::endl;
        std::cout << "max_order set to MAX_MONO_ORDER = " << MAX_MONO_ORDER << std::endl;
        max_order = MAX_MONO_ORDER;
    }

    #ifdef MYDEBUGPRINT
    std::cout << "long_range_real_energy_mono order: " << max_order << std::endl;
    #endif

    coeffs.resize(dof * max_order, 0.0);

    AlignedVector<double> coeffs_tmp(dof * max_order);
    int nnodes = (int)max_order*1.75;
    cheb_nodes_1d(nnodes, nodes, 0, 1);
    fn_v.resize(dof * nnodes);
    for (int idof = 0; idof < dof; idof++) {
        for (int i = 0; i < nnodes; i++) {
            fn_v[idof * nnodes + i] = f(c0, c, nodes[i]);
        }
    }
    monomial_interp_1d(max_order, nnodes, fn_v, coeffs_tmp);

    for (size_t i = 0; i < coeffs_tmp.size(); i++) {
        coeffs[i] = coeffs_tmp[i];
    }
}

void long_range_real_energy(double tol, double tol_coeff, AlignedVector<double>& coeffs, double& c, double& c0) {
    prolc180(tol, c);

    c0 = prolate0_int_eval(c, 1.0);

    int order = MAX_CHEB_ORDER;
    AlignedVector<double> nodes;
    cheb_nodes_1d(order, nodes);

    auto f = [](double c0, double c, double x) {
        double val = prolate0_int_eval(c, x) / c0;
        val = 1 - val;
        return val;
    };
    int dof = 1;
    AlignedVector<double> fn_v(dof * order);
    for (int idof = 0; idof < dof; idof++) {
        for (int i = 0; i < order; i++) {
            fn_v[idof * order + i] = f(c0, c, nodes[i]);
        }
    }

    AlignedVector<double> cheb_coeff;
    cheb_interp_1d(order, fn_v, cheb_coeff);
    int max_order = -1;
    // filter chebyshev coefficients
    AlignedVector<double> max_coeffs(dof, 0.0);
    for (int idof = 0; idof < dof; idof++) {
        for (int i = 0; i < order; i++) {
            max_coeffs[idof] =
                std::max<double>(max_coeffs[idof], std::abs(cheb_coeff[idof * order + i]));
        }
        for (int i = 0; i < order; i++) {
            if (std::abs(cheb_coeff[idof * order + i]) < tol_coeff * max_coeffs[idof]) {
                cheb_coeff[idof * order + i] = 0;
            } else {
                max_order = std::max<int>(max_order, i + 1);
            }
        }
    }

    AlignedVector<double> mono_coeff;
    cheb2mono(order, dof, cheb_coeff, mono_coeff);

    AlignedVector<double>& mono_coeff_filtered = coeffs;
    if (mono_coeff_filtered.size() != dof * max_order) mono_coeff_filtered.resize(dof * max_order);
    for (int i = 0; i < dof; i++) {
        for (int j = 0; j < max_order; j++) {
            mono_coeff_filtered[i * max_order + j] = mono_coeff[i * order + j];
        }
    }
}

void long_range_real_force_cheb(double tol, double tol_coeff, AlignedVector<real>& coeffs) {
    double c;
    prolc180(tol, c);

    double c0 = prolate0_int_eval(c, 1.0);

    int order = MAX_CHEB_ORDER;
    AlignedVector<double> nodes;
    cheb_nodes_1d(order, nodes);

    auto f = [](double c0, double c, double x) {
        //double val = prolate0_int_eval(c, x) / c0 - x / c0 * prolate0_eval(c, x);
        //val = 1 - val;
        double val = x / c0 * prolate0_eval(c, x) - prolate0_int_eval(c, x) / c0;
        return val;
    };
    int dof = 1;
    AlignedVector<double> fn_v(dof * order);
    for (int idof = 0; idof < dof; idof++) {
        for (int i = 0; i < order; i++) {
            fn_v[idof * order + i] = f(c0, c, nodes[i]);
        }
    }

    AlignedVector<double> cheb_coeff;
    cheb_interp_1d(order, fn_v, cheb_coeff);
    int max_order = -1;
    // filter chebyshev coefficients
    double max_coeffs = 0.0;
    for (int idof = 0; idof < dof; idof++) {
        max_coeffs = 0.0;
        for (int i = 0; i < order; i++) {
            max_coeffs = std::max<double>(max_coeffs, std::abs(cheb_coeff[idof * order + i]));
        }
        for (int i = 0; i < order; i++) {
            if (std::abs(cheb_coeff[idof * order + i]) > tol_coeff * max_coeffs) {
                max_order = std::max<int>(max_order, i + 1);
            }
        }
    }

    AlignedVector<real>& cheb_coeff_filtered = coeffs;
    if (cheb_coeff_filtered.size() != dof * max_order) cheb_coeff_filtered.resize(dof * max_order);
    for (int i = 0; i < dof; i++) {
        for (int j = 0; j < max_order; j++) {
            cheb_coeff_filtered[i * max_order + j] = cheb_coeff[i * order + j];
        }
    }
}

void long_range_real_force_mono(double tol, double tol_coeff, AlignedVector<real>& coeffs) {
    double c;
    prolc180(tol, c);

    double c0 = prolate0_int_eval(c, 1.0);

    int order = MAX_CHEB_ORDER;
    AlignedVector<double> nodes;
    cheb_nodes_1d(order, nodes);

    auto f = [](double c0, double c, double x) {
        //double val = prolate0_int_eval(c, x) / c0 - x / c0 * prolate0_eval(c, x);
        //val = 1 - val;
        double val = x / c0 * prolate0_eval(c, x) - prolate0_int_eval(c, x) / c0;
        return val;
    };
    int dof = 1;
    AlignedVector<double> fn_v(dof * order);
    for (int idof = 0; idof < dof; idof++) {
        for (int i = 0; i < order; i++) {
            fn_v[idof * order + i] = f(c0, c, nodes[i]);
        }
    }

    AlignedVector<double> cheb_coeff;
    cheb_interp_1d(order, fn_v, cheb_coeff);
    int max_order = -1;
    // filter chebyshev coefficients
    double max_coeffs = 0.0;
    for (int idof = 0; idof < dof; idof++) {
        max_coeffs = 0.0;
        for (int i = 0; i < order; i++) {
            max_coeffs = std::max<double>(max_coeffs, std::abs(cheb_coeff[idof * order + i]));
        }
        for (int i = 0; i < order; i++) {
            if (std::abs(cheb_coeff[idof * order + i]) > tol_coeff * max_coeffs) {
                max_order = std::max<int>(max_order, i + 1);
            }
        }
    }

    if(max_order > MAX_MONO_ORDER) {
        std::cout << "max_order = " << max_order << " > MAX_MONO_ORDER = " << MAX_MONO_ORDER << std::endl;
        std::cout << "max_order set to MAX_MONO_ORDER = " << MAX_MONO_ORDER << std::endl;
        max_order = MAX_MONO_ORDER;
    }

    #ifdef MYDEBUGPRINT
    std::cout << "long_range_real_force_mono order: " << max_order << std::endl;
    #endif

    coeffs.resize(dof * max_order, 0.0);

    AlignedVector<double> coeffs_tmp(dof * max_order);
    int nnodes = (int)max_order*1.75;
    cheb_nodes_1d(nnodes, nodes, 0, 1);
    fn_v.resize(dof * nnodes);
    for (int idof = 0; idof < dof; idof++) {
        for (int i = 0; i < nnodes; i++) {
            fn_v[idof * nnodes + i] = f(c0, c, nodes[i]);
        }
    }
    monomial_interp_1d(max_order, nnodes, fn_v, coeffs_tmp);

    for (size_t i = 0; i < coeffs_tmp.size(); i++) {
        coeffs[i] = coeffs_tmp[i];
    }
}

void long_range_real_force(double tol, double tol_coeff, AlignedVector<double>& coeffs) {
    double c;
    prolc180(tol, c);

    double c0 = prolate0_int_eval(c, 1.0);

    int order = MAX_CHEB_ORDER;
    AlignedVector<double> nodes;
    cheb_nodes_1d(order, nodes);

    auto f = [](double c0, double c, double x) {
        double val = prolate0_int_eval(c, x) / c0 - x / c0 * prolate0_eval(c, x);
        val = 1 - val;
        return val;
    };
    int dof = 1;
    AlignedVector<double> fn_v(dof * order);
    for (int idof = 0; idof < dof; idof++) {
        for (int i = 0; i < order; i++) {
            fn_v[idof * order + i] = f(c0, c, nodes[i]);
        }
    }

    AlignedVector<double> cheb_coeff;
    cheb_interp_1d(order, fn_v, cheb_coeff);
    int max_order = -1;
    // filter chebyshev coefficients
    AlignedVector<double> max_coeffs(dof, 0.0);
    for (int idof = 0; idof < dof; idof++) {
        for (int i = 0; i < order; i++) {
            max_coeffs[idof] =
                std::max<double>(max_coeffs[idof], std::abs(cheb_coeff[idof * order + i]));
        }
        for (int i = 0; i < order; i++) {
            if (std::abs(cheb_coeff[idof * order + i]) < tol_coeff * max_coeffs[idof]) {
                cheb_coeff[idof * order + i] = 0;
            } else {
                max_order = std::max<int>(max_order, i + 1);
            }
        }
    }

    AlignedVector<double> mono_coeff;
    cheb2mono(order, dof, cheb_coeff, mono_coeff);

    AlignedVector<double>& mono_coeff_filtered = coeffs;
    if (mono_coeff_filtered.size() != dof * max_order) mono_coeff_filtered.resize(dof * max_order);
    for (int i = 0; i < dof; i++) {
        for (int j = 0; j < max_order; j++) {
            mono_coeff_filtered[i * max_order + j] = mono_coeff[i * order + j];
        }
    }
}

void splitting_function_fourier_space_cheb(double tol, double tol_coeff, AlignedVector<real>& coeffs, double& lambda) {
    double c;
    prolc180(tol, c);

    double c0 = prolate0_int_eval(c, 1.0);

    int quad_npts = 200;
    AlignedVector<double> xs(quad_npts, 0), ws(quad_npts, 0);
    gaussian_quadrature(quad_npts, xs.data(), ws.data());
    lambda = 0.0;
    for (int i = 0; i < quad_npts; i++) {
        lambda += ws[i] * prolate0_eval(c, xs[i]) * std::cos(c * xs[i] * 0.5);
    }
    lambda /= prolate0_eval(c, 0.5);

    int order = MAX_CHEB_ORDER;
    AlignedVector<double> nodes;
    cheb_nodes_1d(order, nodes);

    auto f = [](double lambda, double c, double c0, double x) {
        double val = lambda * prolate0_eval(c, x) / c0;
        return val;
    };
    int dof = 1;
    AlignedVector<double> fn_v(dof * order);
    for (int idof = 0; idof < dof; idof++) {
        for (int i = 0; i < order; i++) {
            fn_v[idof * order + i] = f(lambda, c, c0, nodes[i]);
        }
    }

    AlignedVector<double> cheb_coeff;
    cheb_interp_1d(order, fn_v, cheb_coeff);
    int max_order = -1;
    // filter chebyshev coefficients
    double max_coeffs = 0.0;
    for (int idof = 0; idof < dof; idof++) {
        max_coeffs = 0.0;
        for (int i = 0; i < order; i++) {
            max_coeffs = std::max<double>(max_coeffs, std::abs(cheb_coeff[idof * order + i]));
        }
        for (int i = 0; i < order; i++) {
            if (std::abs(cheb_coeff[idof * order + i]) > tol_coeff * max_coeffs) {
                max_order = std::max<int>(max_order, i + 1);
            }
        }
    }

    AlignedVector<real>& cheb_coeff_filtered = coeffs;
    if (cheb_coeff_filtered.size() != dof * max_order) cheb_coeff_filtered.resize(dof * max_order);
    for (int i = 0; i < dof; i++) {
        for (int j = 0; j < max_order; j++) {
            cheb_coeff_filtered[i * max_order + j] = cheb_coeff[i * order + j];
        }
    }
}

void splitting_function_fourier_space_der_cheb(double tol, double tol_coeff, AlignedVector<real>& coeffs, double& lambda) {
    double c;
    prolc180(tol, c);

    double c0 = prolate0_int_eval(c, 1.0);

    int quad_npts = 200;
    AlignedVector<double> xs(quad_npts, 0), ws(quad_npts, 0);
    gaussian_quadrature(quad_npts, xs.data(), ws.data());
    lambda = 0.0;
    for (int i = 0; i < quad_npts; i++) {
        lambda += ws[i] * prolate0_eval(c, xs[i]) * std::cos(c * xs[i] * 0.5);
    }
    lambda /= prolate0_eval(c, 0.5);

    int order = MAX_CHEB_ORDER;
    AlignedVector<double> nodes;
    cheb_nodes_1d(order, nodes);

    auto f = [](double lambda, double c, double c0, double x) {
        double val = lambda * x * prolate0_eval_derivative(c, x) / c0;
        return val;
    };
    int dof = 1;
    AlignedVector<double> fn_v(dof * order);
    for (int idof = 0; idof < dof; idof++) {
        for (int i = 0; i < order; i++) {
            fn_v[idof * order + i] = f(lambda, c, c0, nodes[i]);
        }
    }

    AlignedVector<double> cheb_coeff;
    cheb_interp_1d(order, fn_v, cheb_coeff);
    int max_order = -1;
    // filter chebyshev coefficients
    double max_coeffs = 0.0;
    for (int idof = 0; idof < dof; idof++) {
        max_coeffs = 0.0;
        for (int i = 0; i < order; i++) {
            max_coeffs = std::max<double>(max_coeffs, std::abs(cheb_coeff[idof * order + i]));
        }
        for (int i = 0; i < order; i++) {
            if (std::abs(cheb_coeff[idof * order + i]) > tol_coeff * max_coeffs) {
                max_order = std::max<int>(max_order, i + 1);
            }
        }
    }

    AlignedVector<real>& cheb_coeff_filtered = coeffs;
    if (cheb_coeff_filtered.size() != dof * max_order) cheb_coeff_filtered.resize(dof * max_order);
    for (int i = 0; i < dof; i++) {
        for (int j = 0; j < max_order; j++) {
            cheb_coeff_filtered[i * max_order + j] = cheb_coeff[i * order + j];
        }
    }
}

void splitting_function_fourier_space(double tol, double tol_coeff, AlignedVector<double>& coeffs, double& lambda) {
    double c;
    prolc180(tol, c);

    double c0 = prolate0_int_eval(c, 1.0);

    int quad_npts = 200;
    AlignedVector<double> xs(quad_npts, 0), ws(quad_npts, 0);
    gaussian_quadrature(quad_npts, xs.data(), ws.data());
    lambda = 0.0;
    for (int i = 0; i < quad_npts; i++) {
        lambda += ws[i] * prolate0_eval(c, xs[i]) * std::cos(c * xs[i] * 0.5);
    }
    lambda /= prolate0_eval(c, 0.5);

    int order = MAX_CHEB_ORDER;
    AlignedVector<double> nodes;
    cheb_nodes_1d(order, nodes);

    auto f = [](double lambda, double c, double c0, double x) {
        double val = lambda * prolate0_eval(c, x) / c0;
        return val;
    };
    int dof = 1;
    AlignedVector<double> fn_v(dof * order);
    for (int idof = 0; idof < dof; idof++) {
        for (int i = 0; i < order; i++) {
            fn_v[idof * order + i] = f(lambda, c, c0, nodes[i]);
        }
    }

    AlignedVector<double> cheb_coeff;
    cheb_interp_1d(order, fn_v, cheb_coeff);
    int max_order = -1;
    // filter chebyshev coefficients
    AlignedVector<double> max_coeffs(dof, 0.0);
    for (int idof = 0; idof < dof; idof++) {
        for (int i = 0; i < order; i++) {
            max_coeffs[idof] =
                std::max<double>(max_coeffs[idof], std::abs(cheb_coeff[idof * order + i]));
        }
        for (int i = 0; i < order; i++) {
            if (std::abs(cheb_coeff[idof * order + i]) < tol_coeff * max_coeffs[idof]) {
                cheb_coeff[idof * order + i] = 0;
            } else {
                max_order = std::max<int>(max_order, i + 1);
            }
        }
    }

    AlignedVector<double> mono_coeff;
    cheb2mono(order, dof, cheb_coeff, mono_coeff);

    AlignedVector<double>& mono_coeff_filtered = coeffs;
    if (mono_coeff_filtered.size() != dof * max_order) mono_coeff_filtered.resize(dof * max_order);
    for (int i = 0; i < dof; i++) {
        for (int j = 0; j < max_order; j++) {
            mono_coeff_filtered[i * max_order + j] = mono_coeff[i * order + j];
        }
    }
}

void splitting_function_cheb(double tol, double tol_coeff, AlignedVector<real>& coeffs, double& c, double& c0, double& psi0) {
    prolc180(tol, c);

    c0 = prolate0_int_eval(c, 1.0);
    psi0 = prolate0_eval(c, 0.0);

    int order = MAX_CHEB_ORDER;
    AlignedVector<double> nodes;
    cheb_nodes_1d(order, nodes);

    auto f = [](double c0, double c, double x) {
        double val = prolate0_int_eval(c, x) / c0;
        return val;
    };
    int dof = 1;
    AlignedVector<double> fn_v(dof * order);
    for (int idof = 0; idof < dof; idof++) {
        for (int i = 0; i < order; i++) {
            fn_v[idof * order + i] = f(c0, c, nodes[i]);
        }
    }

    AlignedVector<double> cheb_coeff;
    cheb_interp_1d(order, fn_v, cheb_coeff);
    int max_order = -1;
    // filter chebyshev coefficients
    double max_coeffs = 0.0;
    for (int idof = 0; idof < dof; idof++) {
        max_coeffs = 0.0;
        for (int i = 0; i < order; i++) {
            max_coeffs = std::max<double>(max_coeffs, std::abs(cheb_coeff[idof * order + i]));
        }
        for (int i = 0; i < order; i++) {
            if (std::abs(cheb_coeff[idof * order + i]) > tol_coeff * max_coeffs) {
                max_order = std::max<int>(max_order, i + 1);
            }
        }
    }

    AlignedVector<real>& cheb_coeff_filtered = coeffs;
    if (cheb_coeff_filtered.size() != dof * max_order) cheb_coeff_filtered.resize(dof * max_order);
    for (int i = 0; i < dof; i++) {
        for (int j = 0; j < max_order; j++) {
            cheb_coeff_filtered[i * max_order + j] = cheb_coeff[i * order + j];
        }
    }
}

void splitting_function(double tol, double tol_coeff, AlignedVector<double>& coeffs, double& c, double& c0, double& psi0) {
    prolc180(tol, c);

    c0 = prolate0_int_eval(c, 1.0);
    psi0 = prolate0_eval(c, 0.0);

    int order = MAX_CHEB_ORDER;
    AlignedVector<double> nodes;
    cheb_nodes_1d(order, nodes);

    auto f = [](double c0, double c, double x) {
        double val = prolate0_int_eval(c, x) / c0;
        return val;
    };
    int dof = 1;
    AlignedVector<double> fn_v(dof * order);
    for (int idof = 0; idof < dof; idof++) {
        for (int i = 0; i < order; i++) {
            fn_v[idof * order + i] = f(c0, c, nodes[i]);
        }
    }

    AlignedVector<double> cheb_coeff;
    cheb_interp_1d(order, fn_v, cheb_coeff);
    int max_order = -1;
    // filter chebyshev coefficients
    AlignedVector<double> max_coeffs(dof, 0.0);
    for (int idof = 0; idof < dof; idof++) {
        for (int i = 0; i < order; i++) {
            max_coeffs[idof] =
                std::max<double>(max_coeffs[idof], std::abs(cheb_coeff[idof * order + i]));
        }
        for (int i = 0; i < order; i++) {
            if (std::abs(cheb_coeff[idof * order + i]) < tol_coeff * max_coeffs[idof]) {
                cheb_coeff[idof * order + i] = 0;
            } else {
                max_order = std::max<int>(max_order, i + 1);
            }
        }
    }

    AlignedVector<double> mono_coeff;
    cheb2mono(order, dof, cheb_coeff, mono_coeff);

    AlignedVector<double>& mono_coeff_filtered = coeffs;
    if (mono_coeff_filtered.size() != dof * max_order) mono_coeff_filtered.resize(dof * max_order);
    for (int i = 0; i < dof; i++) {
        for (int j = 0; j < max_order; j++) {
            mono_coeff_filtered[i * max_order + j] = mono_coeff[i * order + j];
        }
    }
}

double spread_window_ref(double c, int P, int i, double x) {
    double arg = x - P / 2.0 + i;
    arg /= P / 2.0;
    if(std::abs(arg) > 1) {
        return 0;
    }
    double val = prolate0_eval(c, arg);
    return val;
}

double spread_window_der_ref(double c, int P, int i, double x) {
    double arg = x - P / 2.0 + i;
    arg /= P / 2.0;
    if(std::abs(arg) > 1) {
        return 0;
    }
    double val = prolate0_eval_derivative(c, arg)*2.0/P;
    return val;
}

double spread_window_fourier_ref(double c, double x) {
    int quad_npts = 200;
    AlignedVector<double> xs(quad_npts, 0), ws(quad_npts, 0);
    gaussian_quadrature(quad_npts, xs.data(), ws.data());
    double lambda = 0.0;
    for (int i = 0; i < quad_npts; i++) {
        lambda += ws[i] * prolate0_eval(c, xs[i]) * std::cos(c * xs[i] * 0.5);
    }
    lambda /= prolate0_eval(c, 0.5);

    if (std::abs(x) > 1) {
        return 0;
    }

    double val = lambda * prolate0_eval(c, x);
    return val;
}

double splitting_function_fourier_space_ref(double c, double c0, double lambda, double x) {

    if (std::abs(x) > 1) {
        return 0;
    }

    /*
    double c0 = prolate0_int_eval(c, 1.0);

    int quad_npts = 200;
    AlignedVector<double> xs(quad_npts, 0), ws(quad_npts, 0);
    gaussian_quadrature(quad_npts, xs.data(), ws.data());
    double lambda = 0.0;
    for (int i = 0; i < quad_npts; i++) {
        lambda += ws[i] * prolate0_eval(c, xs[i]) * std::cos(c * xs[i] * 0.5);
    }
    lambda /= prolate0_eval(c, 0.5);
    */

    double val = lambda * prolate0_eval(c, x) / c0;
    return val;
}

double splitting_function_fourier_space_der_ref(double c, double c0, double lambda, double x) {

    if (std::abs(x) > 1) {
        return 0;
    }

    /*
    double c0 = prolate0_int_eval(c, 1.0);

    int quad_npts = 200;
    AlignedVector<double> xs(quad_npts, 0), ws(quad_npts, 0);
    gaussian_quadrature(quad_npts, xs.data(), ws.data());
    double lambda = 0.0;
    for (int i = 0; i < quad_npts; i++) {
        lambda += ws[i] * prolate0_eval(c, xs[i]) * std::cos(c * xs[i] * 0.5);
    }
    lambda /= prolate0_eval(c, 0.5);
    */

    double val = lambda * x * prolate0_eval_derivative(c, x) / c0;
    return val;
}

double splitting_function_real_space_ref(double c, double c0, double x) {
    if (std::abs(x) > 1) {
        return 1.0;
    }
    //c0 = prolate0_int_eval(c, 1.0);

    double val = prolate0_int_eval(c, x) / c0;
    return val;
}
// end of approximation functions

/*
// test to generate coefficients
// g++-14 -std=c++17 -O3 -march=native pswf.cpp -lopenblas
int main() {
    auto start = std::chrono::steady_clock::now();

    std::cout << std::scientific << std::setprecision(12);

    // spread window real
    int P = 5;
    double tol = 1.0e-6;
    double tol_coeff = 1.0e-6;
    AlignedVector<double> coeffs;
    spread_window_real_space(P, tol, tol_coeff, coeffs);
    std::cout << "spread window real space coeffs: \n";
    int nfuncs = P;
    for (int i = 0; i < nfuncs; i++) {
        for (int j = 0; j < coeffs.size() / nfuncs; j++) {
            std::cout << coeffs[i * (coeffs.size() / nfuncs) + j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    // spread window fourier
    tol = 5e-3;
    tol_coeff = 5e-3;
    spread_window_fourier_space(tol, tol_coeff, coeffs);
    std::cout << "spread window fourier space coeffs: \n";
    nfuncs = 1;
    for (int i = 0; i < nfuncs; i++) {
        for (int j = 0; j < coeffs.size() / nfuncs; j++) {
            std::cout << coeffs[i * (coeffs.size() / nfuncs) + j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    // long range real energy
    tol = 2e-5;
    tol_coeff = 2e-8;
    long_range_real_energy(tol, tol_coeff, coeffs);
    std::cout << "long range real energy coeffs: \n";
    nfuncs = 1;
    for (int i = 0; i < nfuncs; i++) {
        for (int j = 0; j < coeffs.size() / nfuncs; j++) {
            std::cout << coeffs[i * (coeffs.size() / nfuncs) + j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    // long range real force
    tol = 2e-5;
    tol_coeff = 2e-8;
    long_range_real_force(tol, tol_coeff, coeffs);
    std::cout << "long range real force coeffs: \n";
    nfuncs = 1;
    for (int i = 0; i < nfuncs; i++) {
        for (int j = 0; j < coeffs.size() / nfuncs; j++) {
            std::cout << coeffs[i * (coeffs.size() / nfuncs) + j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    // splitting function fourier space
    tol = 2e-5;
    tol_coeff = 2e-8;
    splitting_function_fourier_space(tol, tol_coeff, coeffs);
    std::cout << "splitting function fourier space coeffs: \n";
    nfuncs = 1;
    for (int i = 0; i < nfuncs; i++) {
        for (int j = 0; j < coeffs.size() / nfuncs; j++) {
            std::cout << coeffs[i * (coeffs.size() / nfuncs) + j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    std::cout << "Elapsed time: " << elapsed_seconds.count() << " s\n";

    return 0;
}
*/
