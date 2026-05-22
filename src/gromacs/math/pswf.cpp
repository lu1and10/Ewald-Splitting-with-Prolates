/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright 2026 The GROMACS Authors
 * and the project initiators Erik Lindahl, Berk Hess and David van der Spoel.
 * Consult the AUTHORS/COPYING files and https://www.gromacs.org for details.
 *
 * GROMACS is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public License
 * as published by the Free Software Foundation; either version 2.1
 * of the License, or (at your option) any later version.
 *
 * GROMACS is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with GROMACS; if not, see
 * https://www.gnu.org/licenses, or write to the Free Software Foundation,
 * Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA.
 *
 * If you want to redistribute modifications to GROMACS, please
 * consider that scientific software is very special. Version
 * control is crucial - bugs must be traceable. We will be happy to
 * consider code for inclusion in the official distribution, but
 * derived work must not be called official GROMACS. Details are found
 * in the README & COPYING files - if they are missing, get the
 * official version at https://www.gromacs.org.
 *
 * To help us fund GROMACS development, we humbly ask that you cite
 * the research papers on the package. Check out https://www.gromacs.org.
 */

/* The Pswf0 Legendre expansion implementation in later commits is derived
 * from FINUFFT src/common/pswf.cpp, Apache-2.0 licensed, with user approval.
 */

#include "gmxpre.h"

#include "gromacs/math/pswf.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace gmx::esp
{
namespace
{

void buildMatrixCoefficients(double              lambda,
                             int                 n,
                             double              c,
                             std::vector<double>* lower,
                             std::vector<double>* diagonal,
                             std::vector<double>* upper)
{
    for (int k = 0; 2 * k <= n + 2; ++k)
    {
        const double order = static_cast<double>(2 * k);

        const double alpha0 = order * (order - 1.0)
                              / ((2.0 * order + 1.0) * (2.0 * order - 1.0));
        const double beta0 =
                ((order + 1.0) * (order + 1.0) / (2.0 * order + 3.0)
                 + order * order / (2.0 * order - 1.0))
                / (2.0 * order + 1.0);
        const double gamma0 = (order + 1.0) * (order + 2.0)
                              / ((2.0 * order + 1.0) * (2.0 * order + 3.0));

        (*lower)[k]    = -c * c * alpha0;
        (*diagonal)[k] = lambda - order * (order + 1.0) - c * c * beta0;
        (*upper)[k]    = -c * c * gamma0;

        if (k != 0)
        {
            (*lower)[k] *= std::sqrt((2.0 * k + 0.5) / (2.0 * k - 1.5));
        }
        (*upper)[k] *= std::sqrt((2.0 * k + 0.5) / (2.0 * k + 2.5));
    }
}

void tridiagonalEigenvalues(int n, std::vector<double>* diagonal, std::vector<double>* offDiagonal)
{
    if (n == 1)
    {
        return;
    }

    for (int i = 1; i < n; ++i)
    {
        (*offDiagonal)[i - 1] = (*offDiagonal)[i];
    }
    (*offDiagonal)[n - 1] = 0.0;

    for (int l = 0; l < n; ++l)
    {
        int iter = 0;
        while (true)
        {
            int m = l;
            for (; m < n - 1; ++m)
            {
                const double scale = std::abs((*diagonal)[m]) + std::abs((*diagonal)[m + 1]);
                if (scale + std::abs((*offDiagonal)[m]) == scale)
                {
                    break;
                }
            }

            if (m == l)
            {
                break;
            }
            if (iter++ == 30)
            {
                throw std::runtime_error("Pswf0: tridiagonal eigensolver failed to converge");
            }

            double g = ((*diagonal)[l + 1] - (*diagonal)[l]) / (2.0 * (*offDiagonal)[l]);
            double r = std::hypot(g, 1.0);
            g = (*diagonal)[m] - (*diagonal)[l]
                + (*offDiagonal)[l] / (g + std::copysign(r, g));

            double sine   = 1.0;
            double cosine = 1.0;
            double p      = 0.0;
            for (int i = m - 1; i >= l; --i)
            {
                const double f = sine * (*offDiagonal)[i];
                const double b = cosine * (*offDiagonal)[i];
                r              = std::hypot(f, g);
                (*offDiagonal)[i + 1] = r;
                if (r == 0.0)
                {
                    (*diagonal)[i + 1] -= p;
                    (*offDiagonal)[m] = 0.0;
                    break;
                }

                sine             = f / r;
                cosine           = g / r;
                g                = (*diagonal)[i + 1] - p;
                r                = ((*diagonal)[i] - g) * sine + 2.0 * cosine * b;
                p                = sine * r;
                (*diagonal)[i + 1] = g + p;
                g                = cosine * r - b;
            }

            if (r == 0.0)
            {
                break;
            }
            (*diagonal)[l] -= p;
            (*offDiagonal)[l] = g;
            (*offDiagonal)[m] = 0.0;
        }

        for (int i = l; (i > 0) && ((*diagonal)[i] < (*diagonal)[i - 1]); --i)
        {
            std::swap((*diagonal)[i], (*diagonal)[i - 1]);
        }
    }
}

void factorTridiagonal(std::vector<double>*       diagonal,
                       const std::vector<double>& lower,
                       const std::vector<double>& upper,
                       int                        n,
                       std::vector<double>*       down,
                       std::vector<double>*       up,
                       std::vector<double>*       inverseDiagonal)
{
    for (int i = 0; i + 1 < n; ++i)
    {
        const double factor = lower[i + 1] / (*diagonal)[i];
        (*diagonal)[i + 1] -= upper[i] * factor;
        (*down)[i]              = factor;
        (*up)[i + 1]            = upper[i] / (*diagonal)[i + 1];
        (*inverseDiagonal)[i + 1] = 1.0 / (*diagonal)[i + 1];
    }
    (*inverseDiagonal)[0] = 1.0 / (*diagonal)[0];
}

void solveFactoredTridiagonal(const std::vector<double>& down,
                              const std::vector<double>& up,
                              const std::vector<double>& inverseDiagonal,
                              int                        n,
                              std::vector<double>*       rhs)
{
    for (int i = 0; i + 1 < n; ++i)
    {
        (*rhs)[i + 1] -= down[i] * (*rhs)[i];
    }

    for (int i = n - 1; i > 0; --i)
    {
        (*rhs)[i - 1] -= (*rhs)[i] * up[i];
        (*rhs)[i] *= inverseDiagonal[i];
    }
    (*rhs)[0] *= inverseDiagonal[0];
}

void buildLegendreCoefficients(int n, double c, std::vector<double>* coefficients)
{
    constexpr double kEigenvalueShift = 1.0e-8;
    constexpr int    kNumIterations   = 4;

    const int dimension = n / 2;
    coefficients->assign(dimension + 3, 1.0);

    std::vector<double> lower(dimension + 2);
    std::vector<double> diagonal(dimension + 2);
    std::vector<double> upper(dimension + 2);
    std::vector<double> down(dimension + 2);
    std::vector<double> up(dimension + 2);
    std::vector<double> inverseDiagonal(dimension + 2);

    buildMatrixCoefficients(0.0, n, c, &lower, &diagonal, &upper);
    tridiagonalEigenvalues(dimension, &diagonal, &lower);

    const double lambda = -diagonal[dimension - 1] + kEigenvalueShift;
    buildMatrixCoefficients(lambda, n, c, &lower, &diagonal, &upper);
    factorTridiagonal(&diagonal, lower, upper, dimension, &down, &up, &inverseDiagonal);

    for (int iter = 0; iter < kNumIterations; ++iter)
    {
        solveFactoredTridiagonal(down, up, inverseDiagonal, dimension, coefficients);

        double norm = 0.0;
        for (int j = 0; j < dimension; ++j)
        {
            norm += (*coefficients)[j] * (*coefficients)[j];
        }
        norm = std::sqrt(norm);
        for (int j = 0; j < dimension; ++j)
        {
            (*coefficients)[j] /= norm;
        }
    }

    int lastSignificant = 0;
    for (int i = 0; i < dimension; ++i)
    {
        if (std::abs((*coefficients)[i]) > 1.0e-16)
        {
            lastSignificant = i;
        }
        (*coefficients)[i] *= std::sqrt(2.0 * i + 0.5);
    }
    coefficients->resize(lastSignificant + 1);
}

double evaluateRaw(const std::vector<double>&              coefficients,
                   const std::vector<std::array<double, 3>>& recurrenceCoefficients,
                   double                                  x)
{
    const double xSquared = x * x;
    double       pjm1     = 0.0;
    double       pjm2     = 1.0;
    double       value    = coefficients[0];

    std::size_t i = 1;
    for (; i + 1 < recurrenceCoefficients.size(); i += 2)
    {
        pjm1 = pjm2 * (xSquared * recurrenceCoefficients[i][0] - recurrenceCoefficients[i][1])
               - pjm1 * recurrenceCoefficients[i][2];
        value += coefficients[i] * pjm1;
        pjm2 = pjm1 * (xSquared * recurrenceCoefficients[i + 1][0]
                       - recurrenceCoefficients[i + 1][1])
               - pjm2 * recurrenceCoefficients[i + 1][2];
        value += coefficients[i + 1] * pjm2;
    }

    for (; i < recurrenceCoefficients.size(); ++i)
    {
        const double p =
                pjm2 * (xSquared * recurrenceCoefficients[i][0] - recurrenceCoefficients[i][1])
                - pjm1 * recurrenceCoefficients[i][2];
        value += coefficients[i] * p;
        pjm1 = pjm2;
        pjm2 = p;
    }

    return value;
}

double integrateNormalizedPswf(const std::vector<double>&              coefficients,
                               const std::vector<std::array<double, 3>>& recurrenceCoefficients,
                               double                                  normalizationAt0)
{
    constexpr int    kIntervals = 2048;
    constexpr double kLower     = -1.0;
    constexpr double kUpper     = 1.0;
    const double     h          = (kUpper - kLower) / kIntervals;

    double sum = 0.0;
    for (int i = 0; i <= kIntervals; ++i)
    {
        const double x      = kLower + i * h;
        const double weight = (i == 0 || i == kIntervals) ? 1.0 : (i % 2 == 0 ? 2.0 : 4.0);
        sum += weight * evaluateRaw(coefficients, recurrenceCoefficients, x) * normalizationAt0;
    }

    return sum * h / 3.0;
}

} // namespace

Pswf0::Pswf0(double c) : c_(c), lambda0_(0.0), normalizationAt0_(1.0)
{
    if (c <= 0.0 || c > 30.0)
    {
        throw std::invalid_argument("Pswf0: c must be in (0, 30]");
    }

    static constexpr std::array<int, 20> cToExpansionOrder = { 48,  64,  80,  92,  106,
                                                               120, 130, 144, 156, 168,
                                                               178, 190, 202, 214, 224,
                                                               236, 248, 258, 268, 280 };

    const int cBucket = static_cast<int>(c / 10.0);
    const int n       = (cBucket < static_cast<int>(cToExpansionOrder.size()))
                              ? cToExpansionOrder[cBucket]
                              : static_cast<int>(c * 1.5);

    buildLegendreCoefficients(n, c, &legendreCoefficients_);

    recurrenceCoefficients_.resize(legendreCoefficients_.size());
    for (std::size_t i = 1; i < recurrenceCoefficients_.size(); ++i)
    {
        const double ell = 2.0 * i - 1.0;
        recurrenceCoefficients_[i][0] =
                ((2.0 * ell - 1.0) * (2.0 * ell + 1.0)) / (ell * (ell + 1.0));
        recurrenceCoefficients_[i][1] =
                ((2.0 * ell + 1.0) * (ell - 1.0) * (ell - 1.0)
                 + ell * ell * (2.0 * ell - 3.0))
                / (ell * (ell + 1.0) * (2.0 * ell - 3.0));
        recurrenceCoefficients_[i][2] =
                ((2.0 * ell + 1.0) * (ell - 1.0) * (ell - 2.0))
                / (ell * (ell + 1.0) * (2.0 * ell - 3.0));
    }

    normalizationAt0_ = 1.0 / evaluateRaw(legendreCoefficients_, recurrenceCoefficients_, 0.0);
    lambda0_ = integrateNormalizedPswf(legendreCoefficients_, recurrenceCoefficients_, normalizationAt0_);
}

double Pswf0::eval(double x) const
{
    if (std::abs(x) > 1.0)
    {
        return 0.0;
    }

    return evaluateRaw(legendreCoefficients_, recurrenceCoefficients_, x) * normalizationAt0_;
}

double Pswf0::evalDerivative(double /*x*/) const
{
    return 0.0;
}

double Pswf0::evalIntegral(double /*upper*/) const
{
    return 0.0;
}

double prolc180(double /*tolerance*/)
{
    return 0.0;
}

double prolc180Der(double /*tolerance*/)
{
    return 0.0;
}

double pswfSplitFunction(const Pswf0& /*psi*/, double /*rcInv*/, double /*x*/)
{
    return 0.0;
}

int estimateOrder(double /*tolerance*/)
{
    return 4;
}

void spreadRealPoly(int /*P*/,
                    int /*P_padded*/,
                    double /*tol*/,
                    double /*r_tol*/,
                    double /*c_w*/,
                    AlignedRealVector* coefs,
                    int*               polyOrderOut)
{
    coefs->clear();
    *polyOrderOut = 0;
}

void spreadFourierPoly(double /*tol*/,
                       double /*r_tol*/,
                       double /*c_w*/,
                       AlignedRealVector* coefs,
                       int*               polyOrderOut)
{
    coefs->clear();
    *polyOrderOut = 0;
}

void shortRangeForcePoly(double, double, double, AlignedRealVector* coefs, int* polyOrderOut)
{
    coefs->clear();
    *polyOrderOut = 0;
}

void shortRangeEnergyPoly(double, double, double, AlignedRealVector* coefs, int* polyOrderOut)
{
    coefs->clear();
    *polyOrderOut = 0;
}

void splitFourierPoly(double, double, double, AlignedRealVector* coefs, int* polyOrderOut)
{
    coefs->clear();
    *polyOrderOut = 0;
}

} // namespace gmx::esp
