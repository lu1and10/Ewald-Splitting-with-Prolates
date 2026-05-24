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
 */

#include "gmxpre.h"

#include <array>
#include <vector>

#include <gtest/gtest.h>

#include "gromacs/mdtypes/interaction_const.h"
#include "gromacs/mdtypes/md_enums.h"
#include "gromacs/nbnxm/nbnxm_simd.h"
#include "gromacs/nbnxm/simd_coulomb_functions.h"
#include "gromacs/simd/simd.h"
#include "gromacs/simd/simd_math.h"
#include "gromacs/utility/alignedallocator.h"
#include "gromacs/utility/real.h"

#if GMX_HAVE_NBNXM_SIMD_2XMM
#    define INCLUDE_KERNELFUNCTION_TABLES
#    include "gromacs/nbnxm/kernels_simd_2xmm/kernels.h"
#    undef INCLUDE_KERNELFUNCTION_TABLES
#endif

namespace gmx::test
{
namespace
{

SimdReal loadLaneValues(const std::vector<real>& values)
{
    AlignedArray<real, GMX_SIMD_REAL_WIDTH> lanes;
    for (int i = 0; i < GMX_SIMD_REAL_WIDTH; ++i)
    {
        lanes[i] = values[i % values.size()];
    }
    return load<SimdReal>(lanes);
}

std::vector<real> storeLaneValues(SimdReal value)
{
    AlignedArray<real, GMX_SIMD_REAL_WIDTH> lanes;
    store(lanes.data(), value);
    return std::vector<real>(lanes.begin(), lanes.end());
}

real evaluatePolynomial(const std::vector<real, gmx::AlignedAllocator<real>>& coefs, int order, real s)
{
    real value = coefs[order - 1];
    for (int i = order - 2; i >= 0; --i)
    {
        value = value * s + coefs[i];
    }
    return value;
}

interaction_const_t makeEspInteractionConst()
{
    interaction_const_t ic;
    ic.coulomb.type        = CoulombInteractionType::Esp;
    ic.coulomb.cutoff      = 2.0_real;
    ic.coulomb.ewaldShift  = 0.125_real;
    ic.esp.cutoff          = 2.0_real;
    ic.esp.selfCoeff       = -0.75_real;
    ic.esp.energyPolyOrder = 3;
    ic.esp.energyPolyCoeff = { 1.0_real, -0.25_real, 0.125_real };
    ic.esp.forcePolyOrder  = 3;
    ic.esp.forcePolyCoeff  = { -1.0_real, 0.5_real, -0.125_real };
    return ic;
}

interaction_const_t makeEspInteractionConstWithPolynomialOrders(const int forceOrder, const int energyOrder)
{
    interaction_const_t ic(makeEspInteractionConst());

    ic.esp.forcePolyOrder = forceOrder;
    ic.esp.forcePolyCoeff.resize(ic.esp.forcePolyOrder);
    for (int i = 0; i < ic.esp.forcePolyOrder; ++i)
    {
        ic.esp.forcePolyCoeff[i] =
                ((i % 2 == 0) ? 1.0_real : -1.0_real) * 0.05_real / static_cast<real>(i + 1);
    }

    ic.esp.energyPolyOrder = energyOrder;
    ic.esp.energyPolyCoeff.resize(ic.esp.energyPolyOrder);
    for (int i = 0; i < ic.esp.energyPolyOrder; ++i)
    {
        ic.esp.energyPolyCoeff[i] = 0.025_real / static_cast<real>((i + 1) * (i + 1));
    }

    return ic;
}

void expectCalculatorMatchesScalarReference(const interaction_const_t& ic)
{
    CoulombCalculator<KernelCoulombType::EwaldAnalytical> calculator(ic);

    const std::vector<real>       rValues       = { 0.25_real, 0.5_real, 1.0_real, 1.75_real };
    const auto                    rV            = loadLaneValues(rValues);
    const std::array<SimdReal, 1> rSquaredV     = { rV * rV };
    const std::array<SimdReal, 1> rInvV         = { inv(rV) };
    const std::array<SimdReal, 1> rInvExclV     = rInvV;
    const std::array<SimdBool, 1> withinCutoffV = { rSquaredV[0]
                                                    < SimdReal(ic.coulomb.cutoff * ic.coulomb.cutoff) };

    const std::array<SimdReal, 1> forceV = calculator.force<1>(rSquaredV, rInvV, rInvExclV, withinCutoffV);
    std::array<SimdReal, 1> forceAndEnergyForceV;
    std::array<SimdReal, 1> correctionEnergyV;
    calculator.forceAndCorrectionEnergy<1>(
            rSquaredV, rInvV, rInvExclV, withinCutoffV, forceAndEnergyForceV, correctionEnergyV);

    const std::vector<real> force               = storeLaneValues(forceV[0]);
    const std::vector<real> forceAndEnergyForce = storeLaneValues(forceAndEnergyForceV[0]);
    const std::vector<real> correctionEnergy    = storeLaneValues(correctionEnergyV[0]);

    for (int lane = 0; lane < GMX_SIMD_REAL_WIDTH; ++lane)
    {
        const real r = rValues[lane % rValues.size()];
        const real s = r / ic.esp.cutoff;

        const real expectedForce =
                evaluatePolynomial(ic.esp.forcePolyCoeff, ic.esp.forcePolyOrder, s) / r + 1.0_real / r;
        const real expectedEnergy =
                evaluatePolynomial(ic.esp.energyPolyCoeff, ic.esp.energyPolyOrder, s) / r
                - ic.coulomb.ewaldShift;

        EXPECT_NEAR(force[lane], expectedForce, 5e-6_real) << "lane=" << lane;
        EXPECT_NEAR(forceAndEnergyForce[lane], expectedForce, 5e-6_real) << "lane=" << lane;
        EXPECT_NEAR(correctionEnergy[lane], expectedEnergy, 5e-6_real) << "lane=" << lane;
    }
}

void expectFixedOrderCalculatorMatchesScalarReference(const interaction_const_t& ic)
{
    CoulombCalculator<KernelCoulombType::EwaldAnalytical> calculator(ic);

    const std::vector<real>       rValues       = { 0.25_real, 0.5_real, 1.0_real, 1.75_real };
    const auto                    rV            = loadLaneValues(rValues);
    const std::array<SimdReal, 1> rSquaredV     = { rV * rV };
    const std::array<SimdReal, 1> rInvV         = { inv(rV) };
    const std::array<SimdReal, 1> rInvExclV     = rInvV;
    const std::array<SimdBool, 1> withinCutoffV = { rSquaredV[0]
                                                    < SimdReal(ic.coulomb.cutoff * ic.coulomb.cutoff) };

    const std::array<SimdReal, 1> fixedOrderForceV =
            calculator.force<6, 1>(rSquaredV, rInvV, rInvExclV, withinCutoffV);
    std::array<SimdReal, 1> forceAndEnergyForceV;
    std::array<SimdReal, 1> correctionEnergyV;
    calculator.forceAndCorrectionEnergy<6, 8, 1>(
            rSquaredV, rInvV, rInvExclV, withinCutoffV, forceAndEnergyForceV, correctionEnergyV);

    const std::vector<real> fixedOrderForce     = storeLaneValues(fixedOrderForceV[0]);
    const std::vector<real> forceAndEnergyForce = storeLaneValues(forceAndEnergyForceV[0]);
    const std::vector<real> correctionEnergy    = storeLaneValues(correctionEnergyV[0]);

    for (int lane = 0; lane < GMX_SIMD_REAL_WIDTH; ++lane)
    {
        const real r = rValues[lane % rValues.size()];
        const real s = r / ic.esp.cutoff;

        const real expectedForce =
                evaluatePolynomial(ic.esp.forcePolyCoeff, ic.esp.forcePolyOrder, s) / r + 1.0_real / r;
        const real expectedEnergy =
                evaluatePolynomial(ic.esp.energyPolyCoeff, ic.esp.energyPolyOrder, s) / r
                - ic.coulomb.ewaldShift;

        EXPECT_NEAR(fixedOrderForce[lane], expectedForce, 5e-6_real) << "lane=" << lane;
        EXPECT_NEAR(forceAndEnergyForce[lane], expectedForce, 5e-6_real) << "lane=" << lane;
        EXPECT_NEAR(correctionEnergy[lane], expectedEnergy, 5e-6_real) << "lane=" << lane;
    }
}

TEST(EspShortRangeCoulombCalculator, SelfEnergyUsesEspSelfCoeff)
{
    const interaction_const_t                             ic(makeEspInteractionConst());
    CoulombCalculator<KernelCoulombType::EwaldAnalytical> calculator(ic);

    EXPECT_NEAR(calculator.selfEnergy(), -ic.esp.selfCoeff, 1e-6_real);
}

TEST(EspShortRangeCoulombCalculator, CompileTimePolynomialDispatchCoversAdaptiveOrders)
{
    using Calculator = CoulombCalculator<KernelCoulombType::EwaldAnalytical>;

    EXPECT_TRUE(Calculator::hasEspCompileTimePolynomialOrder(1));
    EXPECT_TRUE(Calculator::hasEspCompileTimePolynomialOrder(21));
    EXPECT_TRUE(Calculator::hasEspCompileTimePolynomialOrder(24));
    EXPECT_FALSE(Calculator::hasEspCompileTimePolynomialOrder(0));
    EXPECT_FALSE(Calculator::hasEspCompileTimePolynomialOrder(25));
}

TEST(EspShortRangeCoulombCalculator, HighSpecializedOrderPolynomialsMatchScalarReference)
{
    expectCalculatorMatchesScalarReference(makeEspInteractionConstWithPolynomialOrders(24, 21));
}

TEST(EspShortRangeCoulombCalculator, FixedOrderApiMatchesScalarReference)
{
    expectFixedOrderCalculatorMatchesScalarReference(makeEspInteractionConstWithPolynomialOrders(6, 8));
}

TEST(EspShortRangeCoulombCalculator, RuntimeFallbackAboveSpecializedOrderMatchesScalarReference)
{
    expectCalculatorMatchesScalarReference(makeEspInteractionConstWithPolynomialOrders(32, 28));
}

#if GMX_HAVE_NBNXM_SIMD_2XMM
TEST(EspShortRangeCoulombCalculator, KernelSelectorUsesFinalOrderSpecializations)
{
    const int coulkt = static_cast<int>(CoulombKernelType::Ewald);
    const int vdwkt  = vdwktLJCUT_COMBNONE;

    EXPECT_EQ(selectNbnxmKernelNoenerEspSimd2xmm(coulkt, vdwkt, 12),
              nbnxmKernelNoenerEspForceOrderSimd2xmm[0][12 - c_nbnxmEspSpecializedForceOrderMin2xmm][vdwkt]);
    EXPECT_EQ(selectNbnxmKernelNoenerEspSimd2xmm(coulkt, vdwkt, 25),
              nbnxmKernelNoenerEspRuntimeOrderSimd2xmm[0][vdwkt]);

    EXPECT_EQ(selectNbnxmKernelEnerEspSimd2xmm(coulkt, vdwkt, 11, 13),
              nbnxmKernelEnerEspMeasuredOrderPairSimd2xmm[0][2][vdwkt]);
    EXPECT_EQ(selectNbnxmKernelEnerEspSimd2xmm(coulkt, vdwkt, 12, 13),
              nbnxmKernelEnerEspEnergyOrderSimd2xmm[0][13 - c_nbnxmEspSpecializedEnergyOrderMin2xmm][vdwkt]);
    EXPECT_EQ(selectNbnxmKernelEnerEspSimd2xmm(coulkt, vdwkt, 25, 25),
              nbnxmKernelEnerEspRuntimeOrderSimd2xmm[0][vdwkt]);

    EXPECT_EQ(selectNbnxmKernelEnergrpEspSimd2xmm(coulkt, vdwkt, 11, 13),
              nbnxmKernelEnergrpEspMeasuredOrderPairSimd2xmm[0][2][vdwkt]);
    EXPECT_EQ(selectNbnxmKernelEnergrpEspSimd2xmm(coulkt, vdwkt, 12, 13),
              nbnxmKernelEnergrpEspEnergyOrderSimd2xmm[0][13 - c_nbnxmEspSpecializedEnergyOrderMin2xmm][vdwkt]);
    EXPECT_EQ(selectNbnxmKernelEnergrpEspSimd2xmm(coulkt, vdwkt, 25, 25),
              nbnxmKernelEnergrpEspRuntimeOrderSimd2xmm[0][vdwkt]);
}
#endif

TEST(EspShortRangeCoulombCalculator, ForceSubtractsLongRangeCorrection)
{
    const interaction_const_t                             ic(makeEspInteractionConst());
    CoulombCalculator<KernelCoulombType::EwaldAnalytical> calculator(ic);

    const std::vector<real>       rValues       = { 0.25_real, 0.5_real, 1.0_real, 1.75_real };
    const auto                    rV            = loadLaneValues(rValues);
    const std::array<SimdReal, 1> rSquaredV     = { rV * rV };
    const std::array<SimdReal, 1> rInvV         = { inv(rV) };
    const std::array<SimdReal, 1> rInvExclV     = rInvV;
    const std::array<SimdBool, 1> withinCutoffV = { rSquaredV[0]
                                                    < SimdReal(ic.coulomb.cutoff * ic.coulomb.cutoff) };

    const std::array<SimdReal, 1> forceV = calculator.force<1>(rSquaredV, rInvV, rInvExclV, withinCutoffV);
    const std::vector<real> force = storeLaneValues(forceV[0]);

    for (int lane = 0; lane < GMX_SIMD_REAL_WIDTH; ++lane)
    {
        const real r = rValues[lane % rValues.size()];
        const real s = r / ic.esp.cutoff;
        const real expected =
                evaluatePolynomial(ic.esp.forcePolyCoeff, ic.esp.forcePolyOrder, s) / r + 1.0_real / r;
        EXPECT_NEAR(force[lane], expected, 2e-6_real) << "lane=" << lane;
    }
}

TEST(EspShortRangeCoulombCalculator, ExcludedForceSubtractsRemovedOneOverR)
{
    const interaction_const_t                             ic(makeEspInteractionConst());
    CoulombCalculator<KernelCoulombType::EwaldAnalytical> calculator(ic);

    const std::vector<real>       rValues       = { 0.25_real, 0.5_real, 1.0_real, 1.75_real };
    const auto                    rV            = loadLaneValues(rValues);
    const std::array<SimdReal, 1> rSquaredV     = { rV * rV };
    const std::array<SimdReal, 1> rInvV         = { inv(rV) };
    const std::array<SimdReal, 1> rInvExclV     = { SimdReal(0.0_real) };
    const std::array<SimdBool, 1> withinCutoffV = { rSquaredV[0]
                                                    < SimdReal(ic.coulomb.cutoff * ic.coulomb.cutoff) };

    const std::array<SimdReal, 1> forceV = calculator.force<1>(rSquaredV, rInvV, rInvExclV, withinCutoffV);
    const std::vector<real> force = storeLaneValues(forceV[0]);

    for (int lane = 0; lane < GMX_SIMD_REAL_WIDTH; ++lane)
    {
        const real r = rValues[lane % rValues.size()];
        const real s = r / ic.esp.cutoff;
        const real expected = evaluatePolynomial(ic.esp.forcePolyCoeff, ic.esp.forcePolyOrder, s) / r;
        EXPECT_NEAR(force[lane], expected, 2e-6_real) << "lane=" << lane;
    }
}

TEST(EspShortRangeCoulombCalculator, EnergyCorrectionLeavesShortRangePotential)
{
    const interaction_const_t                             ic(makeEspInteractionConst());
    CoulombCalculator<KernelCoulombType::EwaldAnalytical> calculator(ic);

    const std::vector<real>       rValues       = { 0.25_real, 0.5_real, 1.0_real, 1.75_real };
    const auto                    rV            = loadLaneValues(rValues);
    const std::array<SimdReal, 1> rSquaredV     = { rV * rV };
    const std::array<SimdReal, 1> rInvV         = { inv(rV) };
    const std::array<SimdReal, 1> rInvExclV     = rInvV;
    const std::array<SimdBool, 1> withinCutoffV = { rSquaredV[0]
                                                    < SimdReal(ic.coulomb.cutoff * ic.coulomb.cutoff) };

    std::array<SimdReal, 1> forceV;
    std::array<SimdReal, 1> correctionEnergyV;
    calculator.forceAndCorrectionEnergy<1>(
            rSquaredV, rInvV, rInvExclV, withinCutoffV, forceV, correctionEnergyV);
    const std::vector<real> correctionEnergy = storeLaneValues(correctionEnergyV[0]);

    for (int lane = 0; lane < GMX_SIMD_REAL_WIDTH; ++lane)
    {
        const real r = rValues[lane % rValues.size()];
        const real s = r / ic.esp.cutoff;
        const real expected = evaluatePolynomial(ic.esp.energyPolyCoeff, ic.esp.energyPolyOrder, s) / r
                              - ic.coulomb.ewaldShift;
        EXPECT_NEAR(correctionEnergy[lane], expected, 2e-6_real) << "lane=" << lane;
    }
}

} // namespace
} // namespace gmx::test
