/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright 2022- The GROMACS Authors
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

/*! \internal
 * \ingroup __module_nbnxm
 *
 * \brief Defines function for computing Coulomb interaction using SIMD
 *
 * The philosophy is as follows. The flavor of Coulomb interactions types
 * is set using constexpr variables. This is then used to select the appropriate
 * templated class with functions in this files through template specialization.
 * Functions in this file take C-style arrays of SIMD of size \p nR registers
 * as arguments, indicated by a suffix with the letter 'V', which are a list
 * of registers with one for each i-particle (4xM kernels) or
 * pair of i-particles (2xMM kernels) that have LJ interactions.
 * Note that we do not use functions for single SIMD registers because this limits
 * the instruction parallelism that compilers can extract.
 *
 * \author Berk Hess <hess@kth.se>
 */

#ifndef GMX_NBNXM_SIMD_COULOMB_FUNCTIONS_H
#define GMX_NBNXM_SIMD_COULOMB_FUNCTIONS_H

#include "gromacs/math/units.h"
#include "gromacs/mdtypes/interaction_const.h"
#include "gromacs/mdtypes/md_enums.h"
#include "gromacs/simd/simd.h"
#include "gromacs/utility/fatalerror.h"

#include "atomdata.h"

namespace gmx
{

//! List of type of Nbnxm kernel coulomb type implementations
enum class KernelCoulombType
{
    RF,              //!< Reaction-field, also used for plain cut-off
    EwaldAnalytical, //!< Ewald with analytical reciprocal contribution correction
    EwaldTabulated,  //!< Ewald with tabulated reciprocal contribution correction
    None             //!< If FMM implementation has its own Coulomb kernels
};

//! Base Coulomb calculator class, only specializations are used
template<KernelCoulombType coulombType>
class CoulombCalculator;

//! Specialized calculator for RF
template<>
class CoulombCalculator<KernelCoulombType::RF>
{
public:
    inline CoulombCalculator(const interaction_const_t& ic) :
        minusTwoTimesRFCoeff_(-2.0_real * ic.coulomb.reactionFieldCoefficient),
        rfOffset_(ic.coulomb.reactionFieldShift),
        selfEnergy_(0.5_real * ic.coulomb.reactionFieldShift)
    {
    }

    //! Returns the self energy
    inline real selfEnergy() const { return selfEnergy_; }

    //! Returns the force
    template<int nR>
    gmx_inline std::array<SimdReal, nR> force(const std::array<SimdReal, nR>&            rSquaredV,
                                              const std::array<SimdReal, nR> gmx_unused& dummyRInvV,
                                              const std::array<SimdReal, nR>&            rInvExclV,
                                              const std::array<SimdBool, nR> gmx_unused& withinCutoffV)
    {
        return genArr<nR>([&](int i)
                          { return fma(rSquaredV[i], minusTwoTimesRFCoeff_, rInvExclV[i]); });
    }

    //! Computes forces and energies without 1/r term for reaction-field
    template<int nR, std::size_t energySize>
    gmx_inline void forceAndCorrectionEnergy(const std::array<SimdReal, nR>&            rSquaredV,
                                             const std::array<SimdReal, nR> gmx_unused& dummyRInvV,
                                             const std::array<SimdReal, nR>&            rInvExclV,
                                             const std::array<SimdBool, nR> gmx_unused& withinCutoffV,
                                             std::array<SimdReal, nR>&         forceV,
                                             std::array<SimdReal, energySize>& correctionEnergyV)
    {
        forceV = genArr<nR>([&](int i)
                            { return fma(rSquaredV[i], minusTwoTimesRFCoeff_, rInvExclV[i]); });

        const SimdReal factor = SimdReal(0.5_real) * minusTwoTimesRFCoeff_;
        correctionEnergyV = genArr<nR>([&](int i) { return fma(rSquaredV[i], factor, rfOffset_); });
    }

private:
    //! -2 * RF-coeff
    const SimdReal minusTwoTimesRFCoeff_;
    //! The RF potential shift
    const SimdReal rfOffset_;
    //! The reaction-field self-energy
    const real selfEnergy_;
};

//! Specialized calculator for Ewald using an analytic approximation
template<>
class CoulombCalculator<KernelCoulombType::EwaldAnalytical>
{
public:
    inline CoulombCalculator(const interaction_const_t& ic) :
        useEsp_(ic.coulomb.type == CoulombInteractionType::Esp),
        beta_(ic.coulomb.ewaldCoeff),
        betaSquared_(gmx::square(ic.coulomb.ewaldCoeff)),
        espInvCutoff_(ic.esp.cutoff > 0 ? 1.0_real / ic.esp.cutoff : 0.0_real),
        espEwaldShift_(ic.coulomb.ewaldShift),
        espForcePolyCoeff_(ic.esp.forcePolyCoeff.data()),
        espEnergyPolyCoeff_(ic.esp.energyPolyCoeff.data()),
        espForcePolyOrder_(ic.esp.forcePolyOrder),
        espEnergyPolyOrder_(ic.esp.energyPolyOrder),
        selfEnergy_(ic.coulomb.type == CoulombInteractionType::Esp
                            ? -ic.esp.selfCoeff
                            : 0.5_real * ic.coulomb.ewaldCoeff * M_2_SQRTPI) // beta/sqrt(pi)
    {
        if (useEsp_)
        {
            GMX_ASSERT(ic.esp.cutoff > 0, "ESP short-range calculator requires a positive cutoff");
            GMX_ASSERT(espForcePolyOrder_ != 0 && espEnergyPolyOrder_ != 0,
                       "ESP short-range calculator requires populated polynomials");
        }
    }

    //! Returns the self energy
    inline real selfEnergy() const { return selfEnergy_; }

    static constexpr int sc_maxSpecializedEspShortRangePolyOrder = 24;

    //! Returns whether an ESP short-range polynomial order has a compile-time path.
    static bool hasEspCompileTimePolynomialOrder(const int order)
    {
        return order > 0 && order <= sc_maxSpecializedEspShortRangePolyOrder;
    }

    template<int nR>
    gmx_inline std::array<SimdReal, nR> force(const std::array<SimdReal, nR>& rSquaredV,
                                              const std::array<SimdReal, nR>& dummyRInvV,
                                              const std::array<SimdReal, nR>& rInvExclV,
                                              const std::array<SimdBool, nR>& withinCutoffV)
    {
        if (useEsp_)
        {
            return dispatchEspForce<nR>(rSquaredV, dummyRInvV, rInvExclV, withinCutoffV);
        }

        const auto brsqV = genArr<nR>(
                [&](int i) { return betaSquared_ * selectByMask(rSquaredV[i], withinCutoffV[i]); });

        const auto ewcorrV = genArr<nR>([&](int i) { return beta_ * pmeForceCorrection(brsqV[i]); });

        return genArr<nR>([&](int i) { return fma(ewcorrV[i], brsqV[i], rInvExclV[i]); });
    }

    //! Computes the Coulomb force and the correction energy for the Ewald reciprocal part
    template<int nR, std::size_t energySize>
    gmx_inline void forceAndCorrectionEnergy(const std::array<SimdReal, nR>&   rSquaredV,
                                             const std::array<SimdReal, nR>&   rInvV,
                                             const std::array<SimdReal, nR>&   rInvExclV,
                                             const std::array<SimdBool, nR>&   withinCutoffV,
                                             std::array<SimdReal, nR>&         forceV,
                                             std::array<SimdReal, energySize>& correctionEnergyV)
    {
        if (useEsp_)
        {
            forceV = dispatchEspForce<nR>(rSquaredV, rInvV, rInvExclV, withinCutoffV);
            correctionEnergyV =
                    dispatchEspCorrectionEnergy<nR, energySize>(rSquaredV, rInvV, withinCutoffV);
            return;
        }

        const auto brsqV = genArr<nR>(
                [&](int i) { return betaSquared_ * selectByMask(rSquaredV[i], withinCutoffV[i]); });

        const auto ewcorrV = genArr<nR>([&](int i) { return beta_ * pmeForceCorrection(brsqV[i]); });

        forceV = genArr<nR>([&](int i) { return fma(ewcorrV[i], brsqV[i], rInvExclV[i]); });

        correctionEnergyV =
                genArr<nR>([&](int i) { return beta_ * pmePotentialCorrection(brsqV[i]); });

        GMX_UNUSED_VALUE(rInvV);
    }

private:
    template<int polyOrder>
    gmx_inline SimdReal evaluateEspPolynomial(const real* const coefs,
                                              const int         runtimeOrder,
                                              const SimdReal&   s) const
    {
        if constexpr (polyOrder > 0)
        {
            static_assert(polyOrder <= sc_maxSpecializedEspShortRangePolyOrder);
            SimdReal value(coefs[polyOrder - 1]);
            for (int i = polyOrder - 2; i >= 0; --i)
            {
                value = fma(value, s, SimdReal(coefs[i]));
            }
            return value;
        }

        // Negative orders are used only by nonbonded-benchmark to force the
        // runtime Horner path with the same polynomial degree as the
        // compile-time dispatch path.
        const int order = runtimeOrder > 0 ? runtimeOrder : -runtimeOrder;
        SimdReal  value(coefs[order - 1]);
        for (int i = order - 2; i >= 0; --i)
        {
            value = fma(value, s, SimdReal(coefs[i]));
        }
        return value;
    }

    template<int forcePolyOrder, int nR>
    gmx_inline std::array<SimdReal, nR> espForce(const std::array<SimdReal, nR>& rSquaredV,
                                                 const std::array<SimdReal, nR>& rInvV,
                                                 const std::array<SimdReal, nR>& rInvExclV,
                                                 const std::array<SimdBool, nR>& withinCutoffV) const
    {
        return genArr<nR>(
                [&](int i)
                {
                    const SimdReal r          = rSquaredV[i] * rInvV[i];
                    const SimdReal s          = r * espInvCutoff_;
                    const SimdReal correction = evaluateEspPolynomial<forcePolyOrder>(
                            espForcePolyCoeff_, espForcePolyOrder_, s);
                    return selectByMask(fma(correction, rInvV[i], rInvExclV[i]), withinCutoffV[i]);
                });
    }

    template<int nR>
    gmx_inline std::array<SimdReal, nR> dispatchEspForce(const std::array<SimdReal, nR>& rSquaredV,
                                                         const std::array<SimdReal, nR>& rInvV,
                                                         const std::array<SimdReal, nR>& rInvExclV,
                                                         const std::array<SimdBool, nR>& withinCutoffV) const
    {
#define GMX_NBNXM_ESP_FORCE_ORDER_CASE(order) \
    case order: return espForce<order, nR>(rSquaredV, rInvV, rInvExclV, withinCutoffV)

        switch (espForcePolyOrder_)
        {
            GMX_NBNXM_ESP_FORCE_ORDER_CASE(1);
            GMX_NBNXM_ESP_FORCE_ORDER_CASE(2);
            GMX_NBNXM_ESP_FORCE_ORDER_CASE(3);
            GMX_NBNXM_ESP_FORCE_ORDER_CASE(4);
            GMX_NBNXM_ESP_FORCE_ORDER_CASE(5);
            GMX_NBNXM_ESP_FORCE_ORDER_CASE(6);
            GMX_NBNXM_ESP_FORCE_ORDER_CASE(7);
            GMX_NBNXM_ESP_FORCE_ORDER_CASE(8);
            GMX_NBNXM_ESP_FORCE_ORDER_CASE(9);
            GMX_NBNXM_ESP_FORCE_ORDER_CASE(10);
            GMX_NBNXM_ESP_FORCE_ORDER_CASE(11);
            GMX_NBNXM_ESP_FORCE_ORDER_CASE(12);
            GMX_NBNXM_ESP_FORCE_ORDER_CASE(13);
            GMX_NBNXM_ESP_FORCE_ORDER_CASE(14);
            GMX_NBNXM_ESP_FORCE_ORDER_CASE(15);
            GMX_NBNXM_ESP_FORCE_ORDER_CASE(16);
            GMX_NBNXM_ESP_FORCE_ORDER_CASE(17);
            GMX_NBNXM_ESP_FORCE_ORDER_CASE(18);
            GMX_NBNXM_ESP_FORCE_ORDER_CASE(19);
            GMX_NBNXM_ESP_FORCE_ORDER_CASE(20);
            GMX_NBNXM_ESP_FORCE_ORDER_CASE(21);
            GMX_NBNXM_ESP_FORCE_ORDER_CASE(22);
            GMX_NBNXM_ESP_FORCE_ORDER_CASE(23);
            GMX_NBNXM_ESP_FORCE_ORDER_CASE(24);
            default: return espForce<0, nR>(rSquaredV, rInvV, rInvExclV, withinCutoffV);
        }

#undef GMX_NBNXM_ESP_FORCE_ORDER_CASE
    }

    template<int energyPolyOrder, int nR, std::size_t energySize>
    gmx_inline std::array<SimdReal, energySize>
               espCorrectionEnergy(const std::array<SimdReal, nR>& rSquaredV,
                                   const std::array<SimdReal, nR>& rInvV,
                                   const std::array<SimdBool, nR>& withinCutoffV) const
    {
        return genArr<energySize>(
                [&](int i)
                {
                    const SimdReal r = rSquaredV[i] * rInvV[i];
                    const SimdReal s = r * espInvCutoff_;
                    const SimdReal longRangeCorrection =
                            evaluateEspPolynomial<energyPolyOrder>(
                                    espEnergyPolyCoeff_, espEnergyPolyOrder_, s)
                            * rInvV[i];
                    return selectByMask(longRangeCorrection - espEwaldShift_, withinCutoffV[i]);
                });
    }

    template<int nR, std::size_t energySize>
    gmx_inline std::array<SimdReal, energySize>
               dispatchEspCorrectionEnergy(const std::array<SimdReal, nR>& rSquaredV,
                                           const std::array<SimdReal, nR>& rInvV,
                                           const std::array<SimdBool, nR>& withinCutoffV) const
    {
#define GMX_NBNXM_ESP_ENERGY_ORDER_CASE(order) \
    case order: return espCorrectionEnergy<order, nR, energySize>(rSquaredV, rInvV, withinCutoffV)

        switch (espEnergyPolyOrder_)
        {
            GMX_NBNXM_ESP_ENERGY_ORDER_CASE(1);
            GMX_NBNXM_ESP_ENERGY_ORDER_CASE(2);
            GMX_NBNXM_ESP_ENERGY_ORDER_CASE(3);
            GMX_NBNXM_ESP_ENERGY_ORDER_CASE(4);
            GMX_NBNXM_ESP_ENERGY_ORDER_CASE(5);
            GMX_NBNXM_ESP_ENERGY_ORDER_CASE(6);
            GMX_NBNXM_ESP_ENERGY_ORDER_CASE(7);
            GMX_NBNXM_ESP_ENERGY_ORDER_CASE(8);
            GMX_NBNXM_ESP_ENERGY_ORDER_CASE(9);
            GMX_NBNXM_ESP_ENERGY_ORDER_CASE(10);
            GMX_NBNXM_ESP_ENERGY_ORDER_CASE(11);
            GMX_NBNXM_ESP_ENERGY_ORDER_CASE(12);
            GMX_NBNXM_ESP_ENERGY_ORDER_CASE(13);
            GMX_NBNXM_ESP_ENERGY_ORDER_CASE(14);
            GMX_NBNXM_ESP_ENERGY_ORDER_CASE(15);
            GMX_NBNXM_ESP_ENERGY_ORDER_CASE(16);
            GMX_NBNXM_ESP_ENERGY_ORDER_CASE(17);
            GMX_NBNXM_ESP_ENERGY_ORDER_CASE(18);
            GMX_NBNXM_ESP_ENERGY_ORDER_CASE(19);
            GMX_NBNXM_ESP_ENERGY_ORDER_CASE(20);
            GMX_NBNXM_ESP_ENERGY_ORDER_CASE(21);
            GMX_NBNXM_ESP_ENERGY_ORDER_CASE(22);
            GMX_NBNXM_ESP_ENERGY_ORDER_CASE(23);
            GMX_NBNXM_ESP_ENERGY_ORDER_CASE(24);
            default: return espCorrectionEnergy<0, nR, energySize>(rSquaredV, rInvV, withinCutoffV);
        }

#undef GMX_NBNXM_ESP_ENERGY_ORDER_CASE
    }

    //! Whether this calculator evaluates ESP instead of standard Ewald.
    const bool useEsp_;
    //! Ewald beta
    const SimdReal beta_;
    //! Ewald beta^2
    const SimdReal betaSquared_;
    //! 1 / ESP short-range cutoff.
    const SimdReal espInvCutoff_;
    //! Existing Ewald shift, compensated before the outer kernel adds it.
    const SimdReal espEwaldShift_;
    //! ESP short-range force polynomial coefficients.
    const real* const espForcePolyCoeff_;
    //! ESP short-range energy polynomial coefficients.
    const real* const espEnergyPolyCoeff_;
    //! ESP short-range force polynomial order.
    const int espForcePolyOrder_;
    //! ESP short-range energy polynomial order.
    const int espEnergyPolyOrder_;
    //! The self energy of the reciprocal part
    const real selfEnergy_;
};

//! Specialized calculator for Ewald using tabulated functions
template<>
class CoulombCalculator<KernelCoulombType::EwaldTabulated>
{
public:
    inline CoulombCalculator(const interaction_const_t& ic) :
        invTableSpacing_(ic.coulombEwaldTables->scale),
        minusHalfTableSpacing_(-0.5_real / ic.coulombEwaldTables->scale),
        tableForce_(c_useTableFormatFDV0 ? ic.coulombEwaldTables->tableFDV0.data()
                                         : ic.coulombEwaldTables->tableF.data()),
        tablePotential_(ic.coulombEwaldTables->tableV.data()),
        selfEnergy_(0.5_real
                    * (c_useTableFormatFDV0 ? ic.coulombEwaldTables->tableFDV0[2]
                                            : ic.coulombEwaldTables->tableV[0]))
    {
    }

    //! Returns the self energy
    inline real selfEnergy() const { return selfEnergy_; }

    //! Returns the force
    template<int nR>
    gmx_inline std::array<SimdReal, nR> force(const std::array<SimdReal, nR>& rSquaredV,
                                              const std::array<SimdReal, nR>& rInvV,
                                              const std::array<SimdReal, nR>& rInvExclV,
                                              const std::array<SimdBool, nR> gmx_unused& withinCutoffV)
    {
        /* We use separate registers for r for tabulated Ewald and LJ to keep the code simpler */
        const auto rV = genArr<nR>([&](int i) { return rSquaredV[i] * rInvV[i]; });

        /* Convert r to scaled table units */
        const auto rScaledV = genArr<nR>([&](int i) { return rV[i] * invTableSpacing_; });

        /* Truncate scaled r to an int */
        std::array<SimdInt32, nR> tableIndexV;
        for (int i = 0; i < nR; i++)
        {
            tableIndexV[i] = cvttR2I(rScaledV[i]);
        }

        /* Convert r to scaled table units */
        const auto rScaledTruncatedV = genArr<nR>([&](int i) { return trunc(rScaledV[i]); });
        const auto rScaledFractionV =
                genArr<nR>([&](int i) { return rScaledV[i] - rScaledTruncatedV[i]; });

        /* Load and interpolate table forces and possibly energies.
         * Force and energy can be combined in one table, stride 4: FDV0
         * or in two separate tables with stride 1: F and V
         * Currently single precision uses FDV0, double F and V.
         */
        std::array<SimdReal, nR> coulombTable0V;
        std::array<SimdReal, nR> coulombTable1V;
        if constexpr (c_useTableFormatFDV0)
        {
            for (int i = 0; i < nR; i++)
            {
                gatherLoadBySimdIntTranspose<4>(
                        tableForce_, tableIndexV[i], &coulombTable0V[i], &coulombTable1V[i]);
            }
        }
        else
        {
            for (int i = 0; i < nR; i++)
            {
                gatherLoadUBySimdIntTranspose<1>(
                        tableForce_, tableIndexV[i], &coulombTable0V[i], &coulombTable1V[i]);
            }
            coulombTable1V = genArr<nR>([&](int i) { return coulombTable1V[i] - coulombTable0V[i]; });
        }
        const auto forceCorrectionV = genArr<nR>(
                [&](int i) { return fma(rScaledFractionV[i], coulombTable1V[i], coulombTable0V[i]); });

        const auto forceV =
                genArr<nR>([&](int i) { return fnma(forceCorrectionV[i], rV[i], rInvExclV[i]); });

        return forceV;
    }

    //! Computes the Coulomb force and the Ewald reciprocal pot correction energy
    template<int nR, std::size_t energySize>
    gmx_inline void forceAndCorrectionEnergy(const std::array<SimdReal, nR>& rSquaredV,
                                             const std::array<SimdReal, nR>& rInvV,
                                             const std::array<SimdReal, nR>& rInvExclV,
                                             const std::array<SimdBool, nR> gmx_unused& withinCutoffV,
                                             std::array<SimdReal, nR>&         forceV,
                                             std::array<SimdReal, energySize>& correctionEnergyV)
    {
        /* We use separate registers for r for tabulated Ewald and LJ to keep the code simpler */
        const auto rV = genArr<nR>([&](int i) { return rSquaredV[i] * rInvV[i]; });

        /* Convert r to scaled table units */
        const auto rScaledV = genArr<nR>([&](int i) { return rV[i] * invTableSpacing_; });

        /* Truncate scaled r to an int */
        std::array<SimdInt32, nR> tableIndexV;
        for (int i = 0; i < nR; i++)
        {
            tableIndexV[i] = cvttR2I(rScaledV[i]);
        }

        /* Convert r to scaled table units */
        const auto rScaledTruncatedV = genArr<nR>([&](int i) { return trunc(rScaledV[i]); });
        const auto rScaledFractionV =
                genArr<nR>([&](int i) { return rScaledV[i] - rScaledTruncatedV[i]; });

        /* Load and interpolate table forces and possibly energies.
         * Force and energy can be combined in one table, stride 4: FDV0
         * or in two separate tables with stride 1: F and V
         * Currently single precision uses FDV0, double F and V.
         */
        std::array<SimdReal, nR> coulombTable0V;
        std::array<SimdReal, nR> coulombTable1V;
        std::array<SimdReal, nR> coulombTablePotV;
        std::array<SimdReal, nR> dumV;
        if constexpr (c_useTableFormatFDV0)
        {
            for (int i = 0; i < nR; i++)
            {
                gatherLoadBySimdIntTranspose<4>(tableForce_,
                                                tableIndexV[i],
                                                &coulombTable0V[i],
                                                &coulombTable1V[i],
                                                &coulombTablePotV[i],
                                                &dumV[i]);
            }
        }
        else
        {
            for (int i = 0; i < nR; i++)
            {
                gatherLoadUBySimdIntTranspose<1>(
                        tableForce_, tableIndexV[i], &coulombTable0V[i], &coulombTable1V[i]);
                gatherLoadUBySimdIntTranspose<1>(
                        tablePotential_, tableIndexV[i], &coulombTablePotV[i], &dumV[i]);
            }
            coulombTable1V = genArr<nR>([&](int i) { return coulombTable1V[i] - coulombTable0V[i]; });
        }
        const auto forceCorrectionV = genArr<nR>(
                [&](int i) { return fma(rScaledFractionV[i], coulombTable1V[i], coulombTable0V[i]); });

        forceV = genArr<nR>([&](int i) { return fnma(forceCorrectionV[i], rV[i], rInvExclV[i]); });

        correctionEnergyV = genArr<nR>(
                [&](int i)
                {
                    return fma((minusHalfTableSpacing_ * rScaledFractionV[i]),
                               (coulombTable0V[i] + forceCorrectionV[i]),
                               coulombTablePotV[i]);
                });
    }

private:
    //! 1 / table-spacing
    const SimdReal invTableSpacing_;
    //! -0.5 * table-spacing
    const SimdReal minusHalfTableSpacing_;
    //! The force table, either a list of forces or a list of F / DeltaF / V / 0
    const real* const tableForce_;
    //! The potential table
    const real* const tablePotential_;
    //! The self energy of the reciprocal part
    const real selfEnergy_;
};

// This class contains no computational logic; it exists solely to allow the construction
// of a CoulombCalculator for any choice of electrostatics treatment
template<>
class CoulombCalculator<KernelCoulombType::None>
{
public:
    inline CoulombCalculator(const interaction_const_t&) {}

    //! Never use this method
    inline real selfEnergy() const
    {
        gmx_fatal(FARGS,
                  "CoulombCalculator->selfEnergy() should not be called when NBNxM kernels are not "
                  "used");
        return 0.0;
    }

    //! Never use this method
    template<int nR>
    inline std::array<SimdReal, nR> force(std::array<SimdReal, nR>&,
                                          std::array<SimdReal, nR>&,
                                          std::array<SimdReal, nR>&,
                                          std::array<SimdBool, nR>&)
    {
        gmx_fatal(
                FARGS,
                "CoulombCalculator->force() should not be called when NBNxM kernels are not used");
        return genArr<nR>([&](int) { return 0.0; });
    }

    //! Never use this method
    template<int nR, std::size_t energySize>
    inline void forceAndCorrectionEnergy(std::array<SimdReal, nR>&,
                                         std::array<SimdReal, nR>&,
                                         std::array<SimdReal, nR>&,
                                         std::array<SimdBool, nR>&,
                                         std::array<SimdReal, nR>&,
                                         std::array<SimdReal, energySize>&)
    {
        gmx_fatal(FARGS,
                  "CoulombCalculator->forceAndCorrectionEnergy() should not be called when NBNxM "
                  "kernels are not used");
    }
};

} // namespace gmx

#endif // GMX_NBNXM_SIMD_COULOMB_FUNCTIONS_H
