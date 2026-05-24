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

#include <cmath>

#include <algorithm>
#include <array>
#include <stdexcept>
#include <vector>

#include "gromacs/utility/gmxassert.h"

namespace gmx::esp
{

using RealAlignedVector = std::vector<real, gmx::AlignedAllocator<real>>;

namespace
{

constexpr double c_pi = 3.141592653589793238462643383279502884;

void buildMatrixCoefficients(double               lambda,
                             int                  n,
                             double               c,
                             std::vector<double>* lower,
                             std::vector<double>* diagonal,
                             std::vector<double>* upper)
{
    for (int k = 0; 2 * k <= n + 2; ++k)
    {
        const double order = static_cast<double>(2 * k);

        const double alpha0 = order * (order - 1.0) / ((2.0 * order + 1.0) * (2.0 * order - 1.0));
        const double beta0  = ((order + 1.0) * (order + 1.0) / (2.0 * order + 3.0)
                              + order * order / (2.0 * order - 1.0))
                             / (2.0 * order + 1.0);
        const double gamma0 = (order + 1.0) * (order + 2.0) / ((2.0 * order + 1.0) * (2.0 * order + 3.0));

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
            g = (*diagonal)[m] - (*diagonal)[l] + (*offDiagonal)[l] / (g + std::copysign(r, g));

            double sine   = 1.0;
            double cosine = 1.0;
            double p      = 0.0;
            for (int i = m - 1; i >= l; --i)
            {
                const double f        = sine * (*offDiagonal)[i];
                const double b        = cosine * (*offDiagonal)[i];
                r                     = std::hypot(f, g);
                (*offDiagonal)[i + 1] = r;
                if (r == 0.0)
                {
                    (*diagonal)[i + 1] -= p;
                    (*offDiagonal)[m] = 0.0;
                    break;
                }

                sine               = f / r;
                cosine             = g / r;
                g                  = (*diagonal)[i + 1] - p;
                r                  = ((*diagonal)[i] - g) * sine + 2.0 * cosine * b;
                p                  = sine * r;
                (*diagonal)[i + 1] = g + p;
                g                  = cosine * r - b;
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
        (*down)[i]                = factor;
        (*up)[i + 1]              = upper[i] / (*diagonal)[i + 1];
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

double evaluateRaw(const std::vector<double>&                coefficients,
                   const std::vector<std::array<double, 3>>& recurrenceCoefficients,
                   double                                    x)
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
        pjm2 = pjm1 * (xSquared * recurrenceCoefficients[i + 1][0] - recurrenceCoefficients[i + 1][1])
               - pjm2 * recurrenceCoefficients[i + 1][2];
        value += coefficients[i + 1] * pjm2;
    }

    for (; i < recurrenceCoefficients.size(); ++i)
    {
        const double p = pjm2 * (xSquared * recurrenceCoefficients[i][0] - recurrenceCoefficients[i][1])
                         - pjm1 * recurrenceCoefficients[i][2];
        value += coefficients[i] * p;
        pjm1 = pjm2;
        pjm2 = p;
    }

    return value;
}

double evaluateRawDerivative(const std::vector<double>&                coefficients,
                             const std::vector<std::array<double, 3>>& recurrenceCoefficients,
                             double                                    x)
{
    const double xSquared = x * x;
    double       pjm1     = 0.0;
    double       pjm2     = 1.0;
    double       dPjm1    = 0.0;
    double       dPjm2    = 0.0;
    double       dValue   = 0.0;

    std::size_t i = 1;
    for (; i < recurrenceCoefficients.size(); ++i)
    {
        const double a  = xSquared * recurrenceCoefficients[i][0] - recurrenceCoefficients[i][1];
        const double dA = 2.0 * x * recurrenceCoefficients[i][0];
        const double p  = pjm2 * a - pjm1 * recurrenceCoefficients[i][2];
        const double dP = dPjm2 * a + pjm2 * dA - dPjm1 * recurrenceCoefficients[i][2];
        dValue += coefficients[i] * dP;
        pjm1  = pjm2;
        pjm2  = p;
        dPjm1 = dPjm2;
        dPjm2 = dP;
    }

    return dValue;
}

struct GLNode
{
    double x;
    double weight;
};

constexpr std::array<GLNode, 16> c_gaussLegendreNodes16 = { {
        { -0.989400934991649932, 0.027152459411754095 },
        { -0.944575023073232576, 0.062253523938647893 },
        { -0.865631202387831744, 0.095158511682492785 },
        { -0.755404408355003034, 0.124628971255533872 },
        { -0.617876244402643748, 0.149595988816576732 },
        { -0.458016777657227386, 0.169156519395002538 },
        { -0.281603550779258913, 0.182603415044923589 },
        { -0.095012509837637440, 0.189450610455068496 },
        { 0.095012509837637440, 0.189450610455068496 },
        { 0.281603550779258913, 0.182603415044923589 },
        { 0.458016777657227386, 0.169156519395002538 },
        { 0.617876244402643748, 0.149595988816576732 },
        { 0.755404408355003034, 0.124628971255533872 },
        { 0.865631202387831744, 0.095158511682492785 },
        { 0.944575023073232576, 0.062253523938647893 },
        { 0.989400934991649932, 0.027152459411754095 },
} };

double integrateNormalizedPswf(const std::vector<double>&                coefficients,
                               const std::vector<std::array<double, 3>>& recurrenceCoefficients,
                               double                                    normalizationAt0)
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

std::vector<double> chebNodes(int n)
{
    std::vector<double> nodes(n);
    for (int k = 0; k < n; ++k)
    {
        nodes[k] = std::cos(c_pi * (k + 0.5) / n);
    }
    return nodes;
}

std::vector<double> chebCoefficientsFromSamples(const std::vector<double>& fSamples)
{
    const int           n = static_cast<int>(fSamples.size());
    std::vector<double> chebCoefficients(n, 0.0);
    for (int j = 0; j < n; ++j)
    {
        double sum = 0.0;
        for (int k = 0; k < n; ++k)
        {
            sum += fSamples[k] * std::cos(c_pi * j * (k + 0.5) / n);
        }
        chebCoefficients[j] = (j == 0 ? 1.0 : 2.0) * sum / n;
    }
    return chebCoefficients;
}

int adaptiveOrderFromChebCoefficients(const std::vector<double>& chebCoefficients, double relativeTolerance)
{
    GMX_ASSERT(relativeTolerance > 0.0, "Adaptive PSWF polynomial fit requires positive tolerance");

    double maxCoeff = 0.0;
    for (const double coefficient : chebCoefficients)
    {
        maxCoeff = std::max(maxCoeff, std::abs(coefficient));
    }
    if (maxCoeff == 0.0)
    {
        return 1;
    }

    const double cutoff = relativeTolerance * maxCoeff;
    int          order  = 1;
    for (int i = 0; i < static_cast<int>(chebCoefficients.size()); ++i)
    {
        if (std::abs(chebCoefficients[i]) > cutoff)
        {
            order = i + 1;
        }
    }
    return order;
}

std::vector<double> multiplyByXMinus(const std::vector<double>& polynomial, double root)
{
    std::vector<double> result(polynomial.size() + 1, 0.0);
    for (std::size_t i = 0; i < polynomial.size(); ++i)
    {
        result[i] += -root * polynomial[i];
        result[i + 1] += polynomial[i];
    }
    return result;
}

std::vector<double> samplesToMonomialAtNodes(const std::vector<double>& nodes,
                                             const std::vector<double>& samples)
{
    GMX_ASSERT(nodes.size() == samples.size(),
               "Polynomial interpolation nodes and samples must match");

    const int           n = static_cast<int>(samples.size());
    std::vector<double> dividedDifferences(samples);
    for (int j = 1; j < n; ++j)
    {
        for (int i = n - 1; i >= j; --i)
        {
            dividedDifferences[i] =
                    (dividedDifferences[i] - dividedDifferences[i - 1]) / (nodes[i] - nodes[i - j]);
        }
    }

    std::vector<double> monomial(n, 0.0);
    monomial[0] = dividedDifferences[0];
    std::vector<double> basis{ 1.0 };
    for (int j = 1; j < n; ++j)
    {
        basis = multiplyByXMinus(basis, nodes[j - 1]);
        for (std::size_t i = 0; i < basis.size(); ++i)
        {
            monomial[i] += dividedDifferences[j] * basis[i];
        }
    }
    return monomial;
}

std::vector<double> chebSamplesToMonomial(const std::vector<double>& fSamples)
{
    return samplesToMonomialAtNodes(chebNodes(static_cast<int>(fSamples.size())), fSamples);
}

int truncateToTol(std::vector<double>* monomial, double tol)
{
    int order = static_cast<int>(monomial->size());
    while (order > 1 && std::abs((*monomial)[order - 1]) < tol)
    {
        --order;
    }
    monomial->resize(order);
    return order;
}

double evaluateMonomial(const std::vector<double>& monomial, double x)
{
    double value = monomial.back();
    for (int i = static_cast<int>(monomial.size()) - 2; i >= 0; --i)
    {
        value = value * x + monomial[i];
    }
    return value;
}

double longRangeEnergyCorrectionScalar(const Pswf0& psi, double s)
{
    return pswfSplitFunction(psi, 1.0, s);
}

double longRangeForceCorrectionScalar(const Pswf0& psi, double s)
{
    const double c0 = psi.evalIntegral(1.0);
    return s * psi.eval(s) / c0 - longRangeEnergyCorrectionScalar(psi, s);
}

double fourierLambda(const Pswf0& psi)
{
    constexpr int    kIntervals = 2048;
    constexpr double kLower     = -1.0;
    constexpr double kUpper     = 1.0;
    const double     h          = (kUpper - kLower) / kIntervals;
    const double     c          = psi.c();

    double sum = 0.0;
    for (int i = 0; i <= kIntervals; ++i)
    {
        const double x      = kLower + i * h;
        const double weight = (i == 0 || i == kIntervals) ? 1.0 : (i % 2 == 0 ? 2.0 : 4.0);
        sum += weight * psi.eval(x) * std::cos(0.5 * c * x);
    }

    return (sum * h / 3.0) / psi.eval(0.5);
}

template<typename Function>
std::vector<double> fitScalarMonomialOnInterval(double lower, double upper, int order, const Function& function)
{
    const std::vector<double> nodes = chebNodes(order);
    const double              half  = 0.5 * (upper - lower);
    const double              mid   = 0.5 * (upper + lower);

    std::vector<double> xNodes(order);
    std::vector<double> samples(order);
    for (int i = 0; i < order; ++i)
    {
        xNodes[i]  = mid + half * nodes[i];
        samples[i] = function(xNodes[i]);
    }

    return samplesToMonomialAtNodes(xNodes, samples);
}

template<typename Function>
double denseMaxAbsoluteFitError(double lower, double upper, const std::vector<double>& monomial, const Function& function)
{
    constexpr int c_numDenseSamples = 257;
    double        maxError          = 0.0;
    for (int i = 0; i < c_numDenseSamples; ++i)
    {
        const double fraction = static_cast<double>(i) / static_cast<double>(c_numDenseSamples - 1);
        const double x        = lower + fraction * (upper - lower);
        maxError = std::max(maxError, std::abs(evaluateMonomial(monomial, x) - function(x)));
    }
    return maxError;
}

template<typename Function>
void fitScalarOnIntervalAdaptive(double             lower,
                                 double             upper,
                                 int                maxOrder,
                                 double             tol,
                                 double             coefficientRelativeTol,
                                 const Function&    function,
                                 RealAlignedVector* coefs,
                                 int*               polyOrderOut)
{
    GMX_ASSERT(maxOrder > 0, "Adaptive PSWF polynomial fit requires positive maximum order");
    GMX_ASSERT(tol > 0.0, "Adaptive PSWF polynomial fit requires positive requested tolerance");
    GMX_ASSERT(coefficientRelativeTol > 0.0,
               "Adaptive PSWF polynomial fit requires positive coefficient tolerance");

    const double              half       = 0.5 * (upper - lower);
    const double              mid        = 0.5 * (upper + lower);
    const std::vector<double> probeNodes = chebNodes(maxOrder);

    std::vector<double> samples(maxOrder);
    for (int i = 0; i < maxOrder; ++i)
    {
        samples[i] = function(mid + half * probeNodes[i]);
    }

    const int order = adaptiveOrderFromChebCoefficients(chebCoefficientsFromSamples(samples),
                                                        coefficientRelativeTol);

    std::vector<double> monomialInX;
    int                 acceptedOrder = 0;
    for (int candidateOrder = order; candidateOrder <= maxOrder; ++candidateOrder)
    {
        monomialInX = fitScalarMonomialOnInterval(lower, upper, candidateOrder, function);
        if (denseMaxAbsoluteFitError(lower, upper, monomialInX, function) <= tol)
        {
            acceptedOrder = candidateOrder;
            break;
        }
    }

    GMX_RELEASE_ASSERT(acceptedOrder > 0,
                       "Adaptive PSWF polynomial fit did not meet requested tolerance");

    coefs->assign(acceptedOrder, 0.0);
    for (int j = 0; j < acceptedOrder; ++j)
    {
        (*coefs)[j] = static_cast<real>(monomialInX[j]);
    }
    *polyOrderOut = acceptedOrder;
}

struct Prolc180Calibration
{
    double K;

    static const Prolc180Calibration& instance()
    {
        static const Prolc180Calibration instance = []
        {
            constexpr std::array<double, 4> calibrationC = { 5.0, 10.0, 15.0, 20.0 };
            double                          logKSum      = 0.0;
            for (double c : calibrationC)
            {
                const Pswf0  psi(c);
                const double psiAtOne = std::abs(psi.eval(1.0));
                const double kLocal   = psiAtOne / (std::sqrt(c) * std::exp(-c));
                logKSum += std::log(kLocal);
            }

            Prolc180Calibration out;
            out.K = std::exp(logKSum / calibrationC.size());
            return out;
        }();
        return instance;
    }
};

} // namespace

Pswf0::Pswf0(double c) : c_(c), lambda0_(0.0), normalizationAt0_(1.0)
{
    if (c <= 0.0 || c > 30.0)
    {
        throw std::invalid_argument("Pswf0: c must be in (0, 30]");
    }

    static constexpr std::array<int, 20> cToExpansionOrder = { 48,  64,  80,  92,  106, 120, 130,
                                                               144, 156, 168, 178, 190, 202, 214,
                                                               224, 236, 248, 258, 268, 280 };

    const int cBucket = static_cast<int>(c / 10.0);
    const int n = (cBucket < static_cast<int>(cToExpansionOrder.size())) ? cToExpansionOrder[cBucket]
                                                                         : static_cast<int>(c * 1.5);

    buildLegendreCoefficients(n, c, &legendreCoefficients_);

    recurrenceCoefficients_.resize(legendreCoefficients_.size());
    for (std::size_t i = 1; i < recurrenceCoefficients_.size(); ++i)
    {
        const double ell = 2.0 * i - 1.0;
        recurrenceCoefficients_[i][0] = ((2.0 * ell - 1.0) * (2.0 * ell + 1.0)) / (ell * (ell + 1.0));
        recurrenceCoefficients_[i][1] =
                ((2.0 * ell + 1.0) * (ell - 1.0) * (ell - 1.0) + ell * ell * (2.0 * ell - 3.0))
                / (ell * (ell + 1.0) * (2.0 * ell - 3.0));
        recurrenceCoefficients_[i][2] = ((2.0 * ell + 1.0) * (ell - 1.0) * (ell - 2.0))
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

double Pswf0::evalDerivative(double x) const
{
    if (std::abs(x) > 1.0)
    {
        return 0.0;
    }

    return evaluateRawDerivative(legendreCoefficients_, recurrenceCoefficients_, x) * normalizationAt0_;
}

double Pswf0::evalIntegral(double upper) const
{
    if (upper == 0.0)
    {
        return 0.0;
    }

    const double sign  = (upper < 0.0) ? -1.0 : 1.0;
    const double limit = std::min(std::abs(upper), 1.0);
    const double half  = 0.5 * limit;

    double sum = 0.0;
    for (const auto& node : c_gaussLegendreNodes16)
    {
        sum += node.weight * eval(half * (node.x + 1.0));
    }

    return sign * half * sum;
}

double prolc180(double tolerance)
{
    if (tolerance <= 0.0 || tolerance >= 1.0)
    {
        throw std::invalid_argument("prolc180: tolerance must be in (0, 1)");
    }

    const double k = Prolc180Calibration::instance().K;
    double       c = std::log(1.0 / tolerance) + 0.5 * std::log(std::log(1.0 / tolerance) + 1.0);

    for (int iter = 0; iter < 50; ++iter)
    {
        const double f     = c - 0.5 * std::log(c) - std::log(k / tolerance);
        const double fPrim = 1.0 - 0.5 / c;
        const double delta = f / fPrim;
        c -= delta;
        if (std::abs(delta) < 1e-10 * std::abs(c))
        {
            break;
        }
    }

    GMX_ASSERT(c > 0.0 && c < 30.0, "prolc180 result out of expected range");
    return c;
}

double prolc180Der(double tolerance)
{
    if (tolerance <= 0.0 || tolerance >= 1.0)
    {
        throw std::invalid_argument("prolc180Der: tolerance must be in (0, 1)");
    }

    const double inverseLog = 1.0 / std::log(1.0 / tolerance);
    return -(1.0 + 0.5 * inverseLog) / tolerance;
}

double pswfSplitFunction(const Pswf0& psi, double rcInv, double x)
{
    if (x <= 0.0)
    {
        return 0.0;
    }

    const double normalizedX = x * rcInv;
    if (normalizedX >= 1.0)
    {
        return 1.0;
    }

    return (2.0 / psi.lambda0()) * psi.evalIntegral(normalizedX);
}

int estimateOrder(double tolerance)
{
    if (tolerance <= 0.0 || tolerance >= 1.0)
    {
        throw std::invalid_argument("estimateOrder: tolerance must be in (0, 1)");
    }

    const double p       = -std::log10(tolerance);
    const double rounded = std::round(p);
    int          order   = 0;

    if (std::abs(p - rounded) < 0.2)
    {
        order = 2 * static_cast<int>(rounded) - 2;
    }
    else
    {
        order = 2 * static_cast<int>(std::ceil(p)) - 3;
    }

    order = std::max(order, 4);
    order = std::min(order, 16);
    return order;
}

void spreadRealPoly(int P, int P_padded, double tol, double r_tol, double c_w, RealAlignedVector* coefs, int* polyOrderOut)
{
    (void)r_tol;
    GMX_ASSERT(P > 0 && P_padded >= P, "spreadRealPoly: invalid P or P_padded");
    GMX_ASSERT(c_w > 0.0, "spreadRealPoly: c_w must be positive");

    const Pswf0 psi(c_w);

    constexpr int                    kInitialOrder = 24;
    std::vector<std::vector<double>> perStencilCoefficients(P);
    int                              globalPolyOrder = 0;
    const std::vector<double>        chebNodesInT    = chebNodes(kInitialOrder);
    std::vector<double>              xNodes(kInitialOrder);
    for (int i = 0; i < kInitialOrder; ++i)
    {
        xNodes[i] = 0.5 * (chebNodesInT[i] + 1.0);
    }

    for (int k = 0; k < P; ++k)
    {
        std::vector<double> samples(kInitialOrder);
        for (int i = 0; i < kInitialOrder; ++i)
        {
            const double x             = xNodes[i];
            const int    tkmBasisIndex = P - k - 1;
            const double s             = (x - 0.5 * static_cast<double>(P) + tkmBasisIndex)
                             / (0.5 * static_cast<double>(P));
            samples[i] = psi.eval(s);
        }

        std::vector<double> monomial = samplesToMonomialAtNodes(xNodes, samples);
        const int           order    = truncateToTol(&monomial, tol);
        globalPolyOrder              = std::max(globalPolyOrder, order);
        perStencilCoefficients[k]    = std::move(monomial);
    }

    coefs->assign(static_cast<std::size_t>(globalPolyOrder) * P_padded, 0.0);
    for (int k = 0; k < P; ++k)
    {
        for (int l = 0; l < static_cast<int>(perStencilCoefficients[k].size()); ++l)
        {
            (*coefs)[l * P_padded + k] = static_cast<real>(perStencilCoefficients[k][l]);
        }
    }
    *polyOrderOut = globalPolyOrder;
}

void spreadFourierPoly(double tol, double r_tol, double c_w, RealAlignedVector* coefs, int* polyOrderOut)
{
    (void)r_tol;
    GMX_ASSERT(c_w > 0.0, "spreadFourierPoly: c_w must be positive");

    const Pswf0  psi(c_w);
    const double lambda = fourierLambda(psi);

    constexpr int             kInitialOrder = 32;
    const std::vector<double> nodes         = chebNodes(kInitialOrder);
    std::vector<double>       sNodes(kInitialOrder);
    std::vector<double>       samples(kInitialOrder);
    for (int i = 0; i < kInitialOrder; ++i)
    {
        sNodes[i]  = 0.5 * (nodes[i] + 1.0);
        samples[i] = lambda * psi.eval(sNodes[i]);
    }

    std::vector<double> monomial = samplesToMonomialAtNodes(sNodes, samples);
    const int           order    = truncateToTol(&monomial, tol);

    coefs->assign(order, 0.0);
    for (int j = 0; j < order; ++j)
    {
        (*coefs)[j] = static_cast<real>(monomial[j]);
    }
    *polyOrderOut = order;
}

void shortRangeForcePoly(double tol, double r_tol, double c, RealAlignedVector* coefs, int* polyOrderOut)
{
    GMX_ASSERT(c > 0.0, "shortRangeForcePoly: c must be positive");

    const Pswf0 psi(c);
    fitScalarOnIntervalAdaptive(
            0.0,
            1.0,
            40,
            tol,
            r_tol,
            [&](double s) { return longRangeForceCorrectionScalar(psi, s); },
            coefs,
            polyOrderOut);
}

void shortRangeEnergyPoly(double tol, double r_tol, double c, RealAlignedVector* coefs, int* polyOrderOut)
{
    GMX_ASSERT(c > 0.0, "shortRangeEnergyPoly: c must be positive");

    const Pswf0 psi(c);
    fitScalarOnIntervalAdaptive(
            0.0,
            1.0,
            40,
            tol,
            r_tol,
            [&](double s) { return longRangeEnergyCorrectionScalar(psi, s); },
            coefs,
            polyOrderOut);
}

void splitFourierPoly(double tol, double r_tol, double c, RealAlignedVector* coefs, int* polyOrderOut)
{
    (void)r_tol;
    GMX_ASSERT(c > 0.0, "splitFourierPoly: c must be positive");

    const Pswf0  psi(c);
    const double c0    = psi.evalIntegral(1.0);
    const double scale = fourierLambda(psi) / c0;

    constexpr int             kOrder = 32;
    const std::vector<double> nodes  = chebNodes(kOrder);
    std::vector<double>       samples(kOrder);
    for (int i = 0; i < kOrder; ++i)
    {
        const double normalizedArg = 0.5 * (nodes[i] + 1.0);
        samples[i]                 = scale * psi.eval(normalizedArg);
    }

    std::vector<double> monomial = chebSamplesToMonomial(samples);
    const int           order    = truncateToTol(&monomial, tol);

    coefs->assign(order, 0.0);
    for (int j = 0; j < order; ++j)
    {
        (*coefs)[j] = static_cast<real>(monomial[j]);
    }
    *polyOrderOut = order;
}

} // namespace gmx::esp
