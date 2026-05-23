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
 * To help us fund GROMACS development, we humbly ask that you cite
 * the research papers on the package. Check out https://www.gromacs.org.
 */

#include "gmxpre.h"

#include <array>
#include <numeric>
#include <vector>

#include <gtest/gtest.h>

#include "gromacs/ewald/pme_internal.h"
#include "gromacs/ewald/pme_spread.h"
#include "gromacs/simd/simd.h"
#include "gromacs/utility/arrayref.h"

namespace gmx::test
{
namespace
{

int paddedOrder(int P)
{
    return ((P + GMX_SIMD_REAL_WIDTH - 1) / GMX_SIMD_REAL_WIDTH) * GMX_SIMD_REAL_WIDTH;
}

EspParameters makePartitionEsp()
{
    EspParameters esp;
    esp.P          = 5;
    esp.P_padded   = paddedOrder(esp.P);
    esp.poly_order = 1;
    esp.rho_coeff.resize(esp.poly_order * esp.P_padded, real(0));

    const std::array<real, 5> weights = { real(0.10), real(0.20), real(0.40), real(0.20), real(0.10) };
    for (int k = 0; k < esp.P; ++k)
    {
        esp.rho_coeff[k] = weights[k];
    }
    return esp;
}

EspParameters makeHornerEsp()
{
    EspParameters esp;
    esp.P          = 5;
    esp.P_padded   = paddedOrder(esp.P);
    esp.poly_order = 4;
    esp.rho_coeff.resize(esp.poly_order * esp.P_padded, real(0));

    for (int order = 0; order < esp.poly_order; ++order)
    {
        for (int k = 0; k < esp.P; ++k)
        {
            esp.rho_coeff[order * esp.P_padded + k] =
                    real(0.01) * real(order + 1) + real(0.02) * real(k + 1);
        }
    }
    return esp;
}

EspParameters makeHornerEsp(int P, int polyOrder)
{
    EspParameters esp;
    esp.P          = P;
    esp.P_padded   = paddedOrder(esp.P);
    esp.poly_order = polyOrder;
    esp.rho_coeff.resize(esp.poly_order * esp.P_padded, real(0));
    esp.drho_coeff.resize(esp.poly_order * esp.P_padded, real(0));

    for (int order = 0; order < esp.poly_order; ++order)
    {
        for (int k = 0; k < esp.P; ++k)
        {
            esp.rho_coeff[order * esp.P_padded + k] =
                    real(0.01) * real(order + 1) + real(0.02) * real(k + 1);
        }
    }

    for (int order = 0; order < esp.poly_order - 1; ++order)
    {
        for (int k = 0; k < esp.P; ++k)
        {
            esp.drho_coeff[order * esp.P_padded + k] =
                    real(order + 1) * esp.rho_coeff[(order + 1) * esp.P_padded + k];
        }
    }
    return esp;
}

std::vector<real> evaluateWithMakePswfs(const EspParameters& esp, real fx, real fy, real fz)
{
    gmx_pme_t pme(nullptr);
    pme.espRuntime = esp;

    std::vector<real> rho(3 * esp.P_padded, real(0));
    make_pswfs(&pme, fx, fy, fz, gmx::makeArrayRef(rho));
    return rho;
}

real scalarHorner(const EspParameters& esp, real x, int k)
{
    real value = esp.rho_coeff[(esp.poly_order - 1) * esp.P_padded + k];
    for (int order = esp.poly_order - 2; order >= 0; --order)
    {
        value = value * x + esp.rho_coeff[order * esp.P_padded + k];
    }
    return value;
}

real scalarDerivativeHorner(const EspParameters& esp, real x, int k)
{
    const int dPolyOrder = esp.poly_order - 1;
    real      value      = esp.drho_coeff[(dPolyOrder - 1) * esp.P_padded + k];
    for (int order = dPolyOrder - 2; order >= 0; --order)
    {
        value = value * x + esp.drho_coeff[order * esp.P_padded + k];
    }
    return value;
}

TEST(EspSpread, ChargeConservation)
{
    const EspParameters     esp = makePartitionEsp();
    const std::vector<real> rho = evaluateWithMakePswfs(esp, real(0.13), real(0.37), real(0.61));

    std::array<real, DIM> dimSums = { real(0), real(0), real(0) };
    for (int dim = 0; dim < DIM; ++dim)
    {
        for (int k = 0; k < esp.P; ++k)
        {
            dimSums[dim] += rho[dim * esp.P_padded + k];
        }
    }

    const real charge = real(2.5);
    EXPECT_NEAR(charge * dimSums[XX] * dimSums[YY] * dimSums[ZZ], charge, real(1e-6));
}

TEST(EspSpread, SimdEqualsScalarReference)
{
    const EspParameters         esp = makeHornerEsp();
    const std::array<real, DIM> x   = { real(0.21), real(0.47), real(0.73) };
    const std::vector<real>     rho = evaluateWithMakePswfs(esp, x[XX], x[YY], x[ZZ]);

    for (int dim = 0; dim < DIM; ++dim)
    {
        for (int k = 0; k < esp.P_padded; ++k)
        {
            EXPECT_NEAR(rho[dim * esp.P_padded + k], scalarHorner(esp, x[dim], k), real(1e-6))
                    << "dim=" << dim << " k=" << k;
        }
    }
}

TEST(EspSpread, PaddingDoesNotPollute)
{
    const EspParameters     esp = makeHornerEsp();
    const std::vector<real> rho = evaluateWithMakePswfs(esp, real(0.25), real(0.50), real(0.75));

    for (int dim = 0; dim < DIM; ++dim)
    {
        for (int k = esp.P; k < esp.P_padded; ++k)
        {
            EXPECT_EQ(rho[dim * esp.P_padded + k], real(0)) << "dim=" << dim << " k=" << k;
        }
    }
}

TEST(EspSpread, CompileTimeDispatchCoverageIncludesEspOrders)
{
    EXPECT_TRUE(make_pswfs_has_compile_time_specialization(4));
    EXPECT_TRUE(make_pswfs_has_compile_time_specialization(5));
    EXPECT_TRUE(make_pswfs_has_compile_time_specialization(6));
    EXPECT_TRUE(make_pswfs_has_compile_time_specialization(7));
    EXPECT_TRUE(make_pswfs_has_compile_time_specialization(8));
    EXPECT_FALSE(make_pswfs_has_compile_time_specialization(9));
}

TEST(EspSpread, CompileTimeDispatchMatchesScalarForEspOrders)
{
    const std::array<real, DIM> x = { real(0.19), real(0.43), real(0.77) };

    for (int P = 4; P <= 8; ++P)
    {
        const EspParameters     esp = makeHornerEsp(P, 6);
        const std::vector<real> rho = evaluateWithMakePswfs(esp, x[XX], x[YY], x[ZZ]);

        for (int dim = 0; dim < DIM; ++dim)
        {
            for (int k = 0; k < esp.P_padded; ++k)
            {
                EXPECT_NEAR(rho[dim * esp.P_padded + k], scalarHorner(esp, x[dim], k), real(1e-6))
                        << "P=" << P << " dim=" << dim << " k=" << k;
            }
        }
    }
}

TEST(EspSpread, RuntimeFallbackMatchesScalarAboveSpecializedPolyOrder)
{
    const EspParameters         esp = makeHornerEsp(5, 17);
    const std::array<real, DIM> x   = { real(0.19), real(0.43), real(0.77) };
    const std::vector<real>     rho = evaluateWithMakePswfs(esp, x[XX], x[YY], x[ZZ]);

    for (int dim = 0; dim < DIM; ++dim)
    {
        for (int k = 0; k < esp.P_padded; ++k)
        {
            EXPECT_NEAR(rho[dim * esp.P_padded + k], scalarHorner(esp, x[dim], k), real(1e-6))
                    << "dim=" << dim << " k=" << k;
        }
    }
}

TEST(EspSpread, EagerDerivativeMatchesScalarReference)
{
    EspParameters esp = makeHornerEsp();
    esp.drho_coeff.resize(esp.poly_order * esp.P_padded, real(0));
    for (int order = 0; order < esp.poly_order - 1; ++order)
    {
        for (int k = 0; k < esp.P; ++k)
        {
            esp.drho_coeff[order * esp.P_padded + k] =
                    real(order + 1) * esp.rho_coeff[(order + 1) * esp.P_padded + k];
        }
    }

    gmx_pme_t pme(nullptr);
    pme.espRuntime = esp;

    std::vector<real> theta(DIM * esp.P_padded, real(0));
    std::vector<real> dtheta(DIM * esp.P_padded, real(0));

    make_pswfs_and_dpswfs(
            &pme, real(0.25), real(0.50), real(0.75), gmx::makeArrayRef(theta), gmx::makeArrayRef(dtheta));
    const std::array<real, DIM> x = { real(0.25), real(0.50), real(0.75) };

    for (int dim = 0; dim < DIM; ++dim)
    {
        for (int k = 0; k < esp.P_padded; ++k)
        {
            EXPECT_NEAR(dtheta[dim * esp.P_padded + k], scalarDerivativeHorner(esp, x[dim], k), real(1e-6))
                    << "dim=" << dim << " k=" << k;
        }
    }
}

} // namespace
} // namespace gmx::test
