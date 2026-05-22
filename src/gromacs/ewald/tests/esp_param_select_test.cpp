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

#include "gmxpre.h"

#include "gromacs/ewald/calculate_spline_moduli.h"
#include "gromacs/ewald/esp_param_select.h"
#include "gromacs/ewald/pme_load_balancing.h"

#include <gtest/gtest.h>

#include "gromacs/math/pswf.h"
#include "gromacs/mdtypes/md_enums.h"
#include "gromacs/mdtypes/simulation_workload.h"
#include "gromacs/utility/logger.h"

#include "testutils/testasserts.h"

namespace gmx::esp::test
{
namespace
{

constexpr real c_pi = 3.14159265358979323846_real;

const gmx::MDLogger nullLogger;

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

EspAutotuneInput makeCubicSpcEWaterInput(real eps)
{
    EspAutotuneInput in{};
    in.accuracy          = eps;
    in.spreadAccuracy    = 0.25_real * eps;
    in.cutoff            = 1.0_real;
    in.box[XX][XX]       = 2.46_real;
    in.box[YY][YY]       = 2.46_real;
    in.box[ZZ][ZZ]       = 2.46_real;
    in.natoms            = 1500;
    in.q2sum             = 500.0 * (0.4238 * 0.4238 * 2 + 0.8476 * 0.8476);
    in.stencilOrderOverride = -1;
    return in;
}

TEST(EspAutotune, ClosedFormBandlimitMatchesProlc180)
{
    const real       eps = 1e-4_real;
    EspAutotuneInput in  = makeCubicSpcEWaterInput(eps);
    EspParameters    out = autotuneEsp(in, nullLogger);

    EXPECT_NEAR(out.c, prolc180(eps), 1.0_real);
    EXPECT_NEAR(out.c1, prolc180(0.5_real * 0.25_real * eps), 1.0_real);
}

TEST(EspAutotune, StencilOrderMatchesPaper3Table2)
{
    EspAutotuneInput in  = makeCubicSpcEWaterInput(1e-4_real);
    EspParameters    out = autotuneEsp(in, nullLogger);

    EXPECT_EQ(out.P, 6);
}

TEST(EspAutotune, StencilOrderMatchesLammpsIntermediateToleranceHeuristic)
{
    EspAutotuneInput in  = makeCubicSpcEWaterInput(3e-5_real);
    EspParameters    out = autotuneEsp(in, nullLogger);

    EXPECT_EQ(out.P, 7);
}

TEST(EspAutotune, GridSpacingMatchesPiRcOverC)
{
    EspAutotuneInput in  = makeCubicSpcEWaterInput(1e-4_real);
    EspParameters    out = autotuneEsp(in, nullLogger);

    ASSERT_GT(out.c, 0);
    ASSERT_GT(out.nx, 0);
    const real expectedSpacing = c_pi * in.cutoff / out.c;
    const real actualSpacingX  = in.box[XX][XX] / out.nx;
    EXPECT_NEAR(actualSpacingX, expectedSpacing, 0.2_real * expectedSpacing);
}

TEST(EspAutotune, TightToleranceUsesPracticalGridSelection)
{
    EspAutotuneInput in = makeCubicSpcEWaterInput(1e-9_real);
    in.box[XX][XX] = 1.0_real;
    in.box[YY][YY] = 1.0_real;
    in.box[ZZ][ZZ] = 1.0_real;

    EspParameters out = autotuneEsp(in, nullLogger);
    EXPECT_GT(out.nx, 1);
    EXPECT_GT(out.ny, 1);
    EXPECT_GT(out.nz, 1);
    EXPECT_LE(out.P, 16);
}

TEST(EspAutotune, UsesGromacsFftGridChooser)
{
    EspAutotuneInput in = makeCubicSpcEWaterInput(1e-4_real);
    const real       h0 = c_pi * in.cutoff / static_cast<real>(prolc180(in.accuracy));
    in.box[XX][XX]      = 13.0_real * h0;
    in.box[YY][YY]      = 13.0_real * h0;
    in.box[ZZ][ZZ]      = 13.0_real * h0;

    EspParameters out = autotuneEsp(in, nullLogger);

    EXPECT_EQ(out.nx, 14);
    EXPECT_EQ(out.ny, 14);
    EXPECT_EQ(out.nz, 14);
}

TEST(EspAutotune, GridRespectsPmeInterpolationMinimum)
{
    EspAutotuneInput in       = makeCubicSpcEWaterInput(1e-5_real);
    in.cutoff                 = 0.8_real;
    in.box[XX][XX]            = 1.86206_real;
    in.box[YY][YY]            = 1.86206_real;
    in.box[ZZ][ZZ]            = 1.86206_real;
    in.stencilOrderOverride   = 12;
    const EspParameters out   = autotuneEsp(in, nullLogger);
    const int           minNx = 2 * (out.P - 1);

    EXPECT_GE(out.nx, minNx);
    EXPECT_GE(out.ny, minNx);
    EXPECT_GE(out.nz, minNx);
}

TEST(EspAutotune, FatalsOnStencilOrderOverrideAboveMax)
{
    EspAutotuneInput in      = makeCubicSpcEWaterInput(1e-4_real);
    in.stencilOrderOverride  = 17;

    GMX_EXPECT_DEATH_IF_SUPPORTED(autotuneEsp(in, nullLogger), "ESP stencil order");
}

TEST(EspAutotune, FatalsOnOversizedGrid)
{
    EspAutotuneInput in = makeCubicSpcEWaterInput(1e-4_real);
    in.box[XX][XX]      = 2000.0_real;
    in.box[YY][YY]      = 2.46_real;
    in.box[ZZ][ZZ]      = 2.46_real;

    GMX_EXPECT_DEATH_IF_SUPPORTED(autotuneEsp(in, nullLogger), "ESP grid is too large");
}

TEST(EspAutotune, FatalsOnInvalidValidatedInput)
{
    EspAutotuneInput in = makeCubicSpcEWaterInput(1e-4_real);
    in.accuracy         = 0;

    GMX_EXPECT_DEATH_IF_SUPPORTED(autotuneEsp(in, nullLogger), "ESP autotune accuracy");
}

TEST(EspAutotune, ScalarFieldsPopulated)
{
    EspAutotuneInput in  = makeCubicSpcEWaterInput(1e-4_real);
    EspParameters    out = autotuneEsp(in, nullLogger);

    EXPECT_NE(out.lambda0, 0.0_real);
    EXPECT_NEAR(out.psi0AtZero, 1.0_real, 1e-6_real);
    EXPECT_NE(out.lambda0_w, 0.0_real);
    EXPECT_LT(out.selfCoeff, 0.0_real);
    EXPECT_EQ(out.cutoff, in.cutoff);
}

TEST(EspAutotune, PolynomialTablesPopulatedRealAndFourier)
{
    EspAutotuneInput in  = makeCubicSpcEWaterInput(1e-4_real);
    EspParameters    out = autotuneEsp(in, nullLogger);

    EXPECT_GT(out.poly_order, 0);
    EXPECT_EQ(out.rho_coeff.size(), static_cast<size_t>(out.poly_order * out.P_padded));
    EXPECT_EQ(out.drho_coeff.size(), out.rho_coeff.size());
    EXPECT_GT(out.spread_fourier_poly_order, 0);
    ASSERT_FALSE(out.spread_fourier_poly.empty());

    const Pswf0  psi(out.c1);
    const double rawWindowAtZero = fourierLambdaReference(psi) * psi.eval(0.0);
    EXPECT_NEAR(out.spread_fourier_poly[0], rawWindowAtZero, 1e-5_real);
}

TEST(EspAutotune, AllPolynomialTablesPopulated)
{
    EspAutotuneInput in  = makeCubicSpcEWaterInput(1e-4_real);
    EspParameters    out = autotuneEsp(in, nullLogger);

    EXPECT_GT(out.split_fourier_poly_order, 0);
    EXPECT_FALSE(out.split_fourier_poly.empty());
    EXPECT_GT(out.short_range_force_poly_order, 0);
    EXPECT_FALSE(out.short_range_force_poly.empty());
    EXPECT_GT(out.short_range_energy_poly_order, 0);
    EXPECT_FALSE(out.short_range_energy_poly.empty());
}

TEST(MakePswfModuli, BspModZeroIndexMatchesRawFourierWindowWithGridScale)
{
    EspAutotuneInput in  = makeCubicSpcEWaterInput(1e-4_real);
    EspParameters    esp = autotuneEsp(in, nullLogger);
    std::array<std::vector<real>, DIM> bspMod;

    make_pswf_moduli(&bspMod, esp, esp.nx, esp.ny, esp.nz);

    const Pswf0  psi(esp.c1);
    const double rawWindowAtZero = fourierLambdaReference(psi) * psi.eval(0.0);
    const real   gridScale       = static_cast<real>(0.5 * esp.P);
    const real   expectedZero    = gridScale * gridScale * static_cast<real>(rawWindowAtZero * rawWindowAtZero);

    EXPECT_EQ(bspMod[XX].size(), static_cast<size_t>(esp.nx));
    EXPECT_EQ(bspMod[YY].size(), static_cast<size_t>(esp.ny));
    EXPECT_EQ(bspMod[ZZ].size(), static_cast<size_t>(esp.nz));
    EXPECT_NEAR(bspMod[XX][0], expectedZero, 1e-5_real);
    EXPECT_NEAR(bspMod[YY][0], expectedZero, 1e-5_real);
    EXPECT_NEAR(bspMod[ZZ][0], expectedZero, 1e-5_real);
}

TEST(MakePswfModuli, SymmetricAroundNyquist)
{
    EspAutotuneInput in  = makeCubicSpcEWaterInput(1e-4_real);
    EspParameters    esp = autotuneEsp(in, nullLogger);
    std::array<std::vector<real>, DIM> bspMod;

    make_pswf_moduli(&bspMod, esp, esp.nx, esp.ny, esp.nz);

    for (int m = 1; m < esp.nx / 2; ++m)
    {
        EXPECT_NEAR(bspMod[XX][m], bspMod[XX][esp.nx - m], 1e-10_real)
                << "Symmetry failed at m=" << m;
    }
}

TEST(EspRuntimeGuards, PmeLoadBalancingIsUnsupportedForEsp)
{
    SimulationWorkload simulationWork;
    simulationWork.useGpuNonbonded = true;

    EXPECT_TRUE(pmeTuningIsSupported(CoulombInteractionType::Pme, false, simulationWork));
    EXPECT_FALSE(pmeTuningIsSupported(CoulombInteractionType::Esp, false, simulationWork));
}

} // namespace
} // namespace gmx::esp::test
