/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright 2026- The GROMACS Authors
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

#include "gmxpre.h"

#include "gromacs/math/pswf.h"

#include <tinyxml2.h>

#include <cmath>

#include <algorithm>
#include <stdexcept>
#include <string>

#include <gtest/gtest.h>

#include "testutils/testfilemanager.h"

namespace gmx::esp::test
{
namespace
{

double longRangeEnergyCorrectionReference(const Pswf0& psi, double r)
{
    return pswfSplitFunction(psi, 1.0, r);
}

double longRangeForceCorrectionReference(const Pswf0& psi, double r)
{
    const double c0  = psi.evalIntegral(1.0);
    const double phi = longRangeEnergyCorrectionReference(psi, r);
    return r * psi.eval(r) / c0 - phi;
}

double fourierLambdaReference(const Pswf0& psi)
{
    constexpr int intervals = 2048;
    const double  c         = psi.c();
    const double  h         = 2.0 / intervals;
    double        sum       = 0.0;
    for (int i = 0; i <= intervals; ++i)
    {
        const double x      = -1.0 + i * h;
        const double weight = (i == 0 || i == intervals) ? 1.0 : (i % 2 == 0 ? 2.0 : 4.0);
        sum += weight * psi.eval(x) * std::cos(0.5 * c * x);
    }
    return (sum * h / 3.0) / psi.eval(0.5);
}

double spreadFourierReference(const Pswf0& psi, double s)
{
    return fourierLambdaReference(psi) * psi.eval(s);
}

double splitFourierReference(const Pswf0& psi, double arg)
{
    const double c0 = psi.evalIntegral(1.0);
    return fourierLambdaReference(psi) * psi.eval(arg / psi.c()) / c0;
}

double splitFourierPolynomialValue(const AlignedRealVector& coefs,
                                   const int                polyOrder,
                                   const double             arg,
                                   const double             bandlimit)
{
    const double x     = 2.0 * arg / bandlimit - 1.0;
    double       value = coefs[polyOrder - 1];
    for (int l = polyOrder - 2; l >= 0; --l)
    {
        value = value * x + coefs[l];
    }
    return value;
}

double polynomialValue(const AlignedRealVector& coefs, const int polyOrder, const double x)
{
    double value = coefs[polyOrder - 1];
    for (int l = polyOrder - 2; l >= 0; --l)
    {
        value = value * x + coefs[l];
    }
    return value;
}

TEST(Pswf0, ConstructionRangeChecks)
{
    EXPECT_THROW(Pswf0(0.0), std::invalid_argument);
    EXPECT_THROW(Pswf0(-1.0), std::invalid_argument);
    EXPECT_THROW(Pswf0(30.01), std::invalid_argument);
    EXPECT_NO_THROW(Pswf0(5.0));
    EXPECT_NO_THROW(Pswf0(12.024));
}

TEST(Pswf0, EvalAtZeroEqualsOneAfterNormalization)
{
    Pswf0 psi(5.0);
    EXPECT_NEAR(psi.eval(0.0), 1.0, 1e-12);
}

TEST(Pswf0, EvalSymmetric)
{
    Pswf0 psi(8.0);
    EXPECT_NEAR(psi.eval(0.3), psi.eval(-0.3), 1e-12);
    EXPECT_NEAR(psi.eval(0.7), psi.eval(-0.7), 1e-12);
}

TEST(Pswf0, EvalReturnsZeroOutsideSupport)
{
    Pswf0 psi(5.0);
    EXPECT_DOUBLE_EQ(psi.eval(1.5), 0.0);
    EXPECT_DOUBLE_EQ(psi.eval(-2.0), 0.0);
}

TEST(Pswf0, MatchesMpmathRefdataAt50Digits)
{
    tinyxml2::XMLDocument doc;
    const auto refPath = gmx::test::TestFileManager::getInputFilePath("refdata/pswf_reference.xml");
    ASSERT_EQ(doc.LoadFile(refPath.string().c_str()), tinyxml2::XML_SUCCESS);

    const auto* root = doc.FirstChildElement("PswfRefdata");
    ASSERT_NE(root, nullptr);

    for (const auto* cNode = root->FirstChildElement("CValue"); cNode != nullptr;
         cNode             = cNode->NextSiblingElement("CValue"))
    {
        double c = 0.0;
        ASSERT_EQ(cNode->QueryDoubleAttribute("c", &c), tinyxml2::XML_SUCCESS);

        Pswf0 psi(c);
        for (const auto* sample = cNode->FirstChildElement("Sample"); sample != nullptr;
             sample             = sample->NextSiblingElement("Sample"))
        {
            double x = 0.0;
            ASSERT_EQ(sample->QueryDoubleAttribute("x", &x), tinyxml2::XML_SUCCESS);
            ASSERT_NE(sample->GetText(), nullptr);

            const double referenceValue = std::stod(sample->GetText());
            EXPECT_NEAR(psi.eval(x), referenceValue, 1e-10) << "c=" << c << " x=" << x;
        }
    }
}

TEST(Pswf0, EvalDerivativeVsFiniteDifference)
{
    Pswf0        psi(8.0);
    const double h = 1e-6;
    for (double x : { -0.7, -0.3, 0.0, 0.3, 0.7 })
    {
        const double finiteDifference = (psi.eval(x + h) - psi.eval(x - h)) / (2.0 * h);
        EXPECT_NEAR(psi.evalDerivative(x), finiteDifference, 1e-7) << "x=" << x;
    }
}

TEST(Pswf0, EvalDerivativeAtZeroIsZero)
{
    Pswf0 psi(8.0);
    EXPECT_NEAR(psi.evalDerivative(0.0), 0.0, 1e-12);
}

TEST(Pswf0, IntegralAtZeroIsZero)
{
    Pswf0 psi(8.0);
    EXPECT_NEAR(psi.evalIntegral(0.0), 0.0, 1e-14);
}

TEST(Pswf0, IntegralAtOneEqualsHalfLambda0OverPsiAtZero)
{
    Pswf0        psi(8.0);
    const double lambda0 = psi.lambda0();
    EXPECT_NEAR(psi.evalIntegral(1.0), 0.5 * lambda0, 1e-9);
}

TEST(Prolc180, MonotoneInTolerance)
{
    const double c3 = prolc180(1e-3);
    const double c4 = prolc180(1e-4);
    const double c5 = prolc180(1e-5);
    EXPECT_LT(c3, c4);
    EXPECT_LT(c4, c5);
}

TEST(Prolc180, MatchesPaper3Table2)
{
    EXPECT_NEAR(prolc180(1e-3), 9.5392, 0.5);
    EXPECT_NEAR(prolc180(5e-4), 10.290, 0.5);
    EXPECT_NEAR(prolc180(1e-4), 12.024, 0.5);
    EXPECT_NEAR(prolc180(5e-5), 12.762, 0.6);
    EXPECT_NEAR(prolc180(1e-5), 14.471, 0.6);
}

TEST(SplitFunction, AtZeroIsZero)
{
    Pswf0 psi(8.0);
    EXPECT_NEAR(pswfSplitFunction(psi, 1.0, 0.0), 0.0, 1e-14);
}

TEST(SplitFunction, AtRcEqualsOne)
{
    Pswf0        psi(8.0);
    const double rc = 1.0;
    EXPECT_NEAR(pswfSplitFunction(psi, 1.0 / rc, rc), 1.0, 1e-9);
}

TEST(SplitFunction, MonotoneIncreasingOn_0_Rc)
{
    Pswf0        psi(8.0);
    const double rc   = 1.0;
    double       last = -1.0;
    for (double r : { 0.1, 0.2, 0.5, 0.8, 0.95 })
    {
        const double value = pswfSplitFunction(psi, 1.0 / rc, r);
        EXPECT_GT(value, last) << "r=" << r;
        last = value;
    }
}

TEST(EstimateOrder, MatchesPaper3Table2)
{
    EXPECT_EQ(estimateOrder(1e-3), 4);
    EXPECT_EQ(estimateOrder(1e-4), 6);
    EXPECT_EQ(estimateOrder(1e-5), 8);
    EXPECT_EQ(estimateOrder(1e-6), 10);
    EXPECT_EQ(estimateOrder(1e-7), 12);
}

TEST(EstimateOrder, MatchesLammpsIntermediateToleranceHeuristic)
{
    EXPECT_EQ(estimateOrder(2e-4), 5);
    EXPECT_EQ(estimateOrder(7e-5), 6);
    EXPECT_EQ(estimateOrder(3e-5), 7);
}

TEST(SpreadRealPoly, AccuracyVsGromacsPmeFractionConvention)
{
    constexpr int    p       = 6;
    constexpr int    pPadded = 8;
    constexpr double cWindow = 12.024;

    AlignedRealVector coefs;
    int               polyOrder = 0;
    spreadRealPoly(p, pPadded, 1e-5, 1e-6, cWindow, &coefs, &polyOrder);

    ASSERT_GT(polyOrder, 0);
    ASSERT_EQ(coefs.size(), static_cast<std::size_t>(polyOrder * pPadded));

    Pswf0 psi(cWindow);
    for (int k = 0; k < p; ++k)
    {
        for (double x : { 0.05, 0.25, 0.5, 0.75, 0.95 })
        {
            const int    tkmBasisIndex = p - k - 1;
            const double s             = (x - 0.5 * static_cast<double>(p) + tkmBasisIndex)
                             / (0.5 * static_cast<double>(p));
            const double ref = psi.eval(s);

            double poly = coefs[(polyOrder - 1) * pPadded + k];
            for (int l = polyOrder - 2; l >= 0; --l)
            {
                poly = poly * x + coefs[l * pPadded + k];
            }
            EXPECT_NEAR(poly, ref, 1e-4) << "k=" << k << " x=" << x;
        }
    }
}

TEST(SpreadRealPoly, PaddedTailIsZero)
{
    constexpr int p       = 5;
    constexpr int pPadded = 8;

    AlignedRealVector coefs;
    int               polyOrder = 0;
    spreadRealPoly(p, pPadded, 1e-4, 1e-5, 9.5392, &coefs, &polyOrder);

    for (int l = 0; l < polyOrder; ++l)
    {
        for (int k = p; k < pPadded; ++k)
        {
            EXPECT_EQ(coefs[l * pPadded + k], 0.0) << "l=" << l << " k=" << k;
        }
    }
}

TEST(SpreadFourierPoly, AtZeroMatchesRawFourierWindow)
{
    AlignedRealVector coefs;
    int               polyOrder = 0;
    spreadFourierPoly(1e-5, 1e-6, 12.024, &coefs, &polyOrder);
    ASSERT_GT(polyOrder, 0);

    const Pswf0 psi(12.024);
    EXPECT_NEAR(coefs[0], spreadFourierReference(psi, 0.0), 1e-5);
}

TEST(SpreadFourierPoly, MatchesRawFourierWindowAt03)
{
    AlignedRealVector coefs;
    int               polyOrder = 0;
    spreadFourierPoly(1e-5, 1e-6, 12.024, &coefs, &polyOrder);
    ASSERT_GT(polyOrder, 0);

    Pswf0        psi(12.024);
    const double s   = 0.3;
    const double ref = spreadFourierReference(psi, s);

    double poly = coefs[polyOrder - 1];
    for (int l = polyOrder - 2; l >= 0; --l)
    {
        poly = poly * s + coefs[l];
    }
    EXPECT_NEAR(poly, ref, 1e-6);
}

TEST(SplitFourierPoly, MatchesChiHatAt05)
{
    AlignedRealVector coefs;
    int               polyOrder = 0;
    splitFourierPoly(1e-5, 1e-6, 12.024, &coefs, &polyOrder);
    ASSERT_GT(polyOrder, 0);

    Pswf0        psi(12.024);
    const double arg = 0.5;
    const double ref = splitFourierReference(psi, arg);

    const double poly = splitFourierPolynomialValue(coefs, polyOrder, arg, psi.c());
    EXPECT_NEAR(poly, ref, 1e-5);
}

TEST(SplitFourierPoly, MatchesLammpsFourierKernelConvention)
{
    constexpr double  c = 12.024;
    AlignedRealVector coefs;
    int               polyOrder = 0;
    splitFourierPoly(1e-5, 1e-6, c, &coefs, &polyOrder);
    ASSERT_GT(polyOrder, 0);

    Pswf0        psi(c);
    const double arg = 0.3 * c;
    const double ref = splitFourierReference(psi, arg);

    const double poly = splitFourierPolynomialValue(coefs, polyOrder, arg, c);
    EXPECT_NEAR(poly, ref, 1e-5);
}

TEST(SplitFourierPoly, MatchesChiHatNearBandlimit)
{
    constexpr double  c = 12.024;
    AlignedRealVector coefs;
    int               polyOrder = 0;
    splitFourierPoly(1e-5, 1e-6, c, &coefs, &polyOrder);
    ASSERT_GT(polyOrder, 0);

    Pswf0        psi(c);
    const double arg = 0.9 * c;
    const double ref = splitFourierReference(psi, arg);

    const double poly = splitFourierPolynomialValue(coefs, polyOrder, arg, c);
    EXPECT_NEAR(poly, ref, 1e-3);
}

TEST(ShortRangeEnergyPoly, MatchesLongRangeCorrectionPhi)
{
    AlignedRealVector coefs;
    int               polyOrder = 0;
    shortRangeEnergyPoly(1e-5, 1e-6, 12.024, &coefs, &polyOrder);
    ASSERT_GT(polyOrder, 0);

    Pswf0 psi(12.024);
    for (double r : { 0.1, 0.3, 0.5, 0.7, 0.9 })
    {
        const double poly = polynomialValue(coefs, polyOrder, r);
        EXPECT_NEAR(poly, longRangeEnergyCorrectionReference(psi, r), 1e-4) << "r=" << r;
    }
}

TEST(ShortRangeForcePoly, MatchesLongRangeForceCorrection)
{
    AlignedRealVector coefs;
    int               polyOrder = 0;
    shortRangeForcePoly(1e-5, 1e-6, 12.024, &coefs, &polyOrder);
    ASSERT_GT(polyOrder, 0);

    Pswf0 psi(12.024);
    for (double r : { 0.1, 0.3, 0.5, 0.7, 0.9 })
    {
        const double poly = polynomialValue(coefs, polyOrder, r);
        EXPECT_NEAR(poly, longRangeForceCorrectionReference(psi, r), 1e-3) << "r=" << r;
    }
}

TEST(ShortRangePoly, TightToleranceRaisesAdaptiveOrderPastLegacyCap)
{
    AlignedRealVector looseCoefs;
    AlignedRealVector tightCoefs;
    int               looseOrder = 0;
    int               tightOrder = 0;

    shortRangeForcePoly(1e-3, 1e-3, 14.471, &looseCoefs, &looseOrder);
    shortRangeForcePoly(1e-8, 1e-10, 14.471, &tightCoefs, &tightOrder);

    EXPECT_LT(looseOrder, tightOrder);
    EXPECT_GT(tightOrder, 16);
}

TEST(ShortRangePoly, TightToleranceMatchesDenseReferenceSamples)
{
    AlignedRealVector forceCoefs;
    AlignedRealVector energyCoefs;
    int               forceOrder  = 0;
    int               energyOrder = 0;

    constexpr double c = 14.471;
    shortRangeForcePoly(1e-8, 1e-10, c, &forceCoefs, &forceOrder);
    shortRangeEnergyPoly(1e-8, 1e-10, c, &energyCoefs, &energyOrder);
    ASSERT_GT(forceOrder, 16);
    ASSERT_GT(energyOrder, 16);

    const Pswf0  psi(c);
    const double tolerance = GMX_DOUBLE ? 5e-8 : 5e-5;
    for (int i = 1; i < 64; ++i)
    {
        const double r = static_cast<double>(i) / 64.0;
        EXPECT_NEAR(polynomialValue(forceCoefs, forceOrder, r), longRangeForceCorrectionReference(psi, r), tolerance)
                << "force r=" << r << " order=" << forceOrder;
        EXPECT_NEAR(polynomialValue(energyCoefs, energyOrder, r), longRangeEnergyCorrectionReference(psi, r), tolerance)
                << "energy r=" << r << " order=" << energyOrder;
    }
}

} // namespace
} // namespace gmx::esp::test
