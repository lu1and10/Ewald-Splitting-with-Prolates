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

#include "gromacs/ewald/pme_gather.h"
#include "gromacs/ewald/pme_internal.h"
#include "gromacs/ewald/pme_spread.h"
#include "gromacs/simd/simd.h"
#include "gromacs/utility/arrayref.h"

#include <array>
#include <vector>

#include <gtest/gtest.h>

namespace gmx::test
{
namespace
{

int paddedOrder(int P)
{
    return ((P + GMX_SIMD_REAL_WIDTH - 1) / GMX_SIMD_REAL_WIDTH) * GMX_SIMD_REAL_WIDTH;
}

EspParameters makeDerivativeEsp()
{
    EspParameters esp;
    esp.P          = 5;
    esp.P_padded   = paddedOrder(esp.P);
    esp.poly_order = 4;
    esp.rho_coeff.resize(esp.poly_order * esp.P_padded, real(0));
    esp.drho_coeff.resize(esp.poly_order * esp.P_padded, real(0));

    for (int order = 0; order < esp.poly_order; ++order)
    {
        for (int k = 0; k < esp.P; ++k)
        {
            esp.rho_coeff[order * esp.P_padded + k] =
                    real(0.03) * real(order + 1) + real(0.015) * real(k + 1);
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

std::vector<real> evaluateSpread(const EspParameters& esp, const std::array<real, DIM>& x)
{
    gmx_pme_t pme(nullptr);
    pme.espRuntime = esp;
    std::vector<real> rho(DIM * esp.P_padded, real(0));
    make_pswfs(&pme, x[XX], x[YY], x[ZZ], gmx::makeArrayRef(rho));
    return rho;
}

std::vector<real> evaluateDerivative(const EspParameters& esp, const std::array<real, DIM>& x)
{
    gmx_pme_t pme(nullptr);
    pme.espRuntime = esp;
    std::vector<real> drho(DIM * esp.P_padded, real(0));
    gather_f_pswfs(&pme, x[XX], x[YY], x[ZZ], gmx::makeArrayRef(drho));
    return drho;
}

real scalarHorner(gmx::ArrayRef<const real> coefs, int order, int Pp, real x, int k)
{
    real value = coefs[(order - 1) * Pp + k];
    for (int i = order - 2; i >= 0; --i)
    {
        value = value * x + coefs[i * Pp + k];
    }
    return value;
}

TEST(EspGather, AdjointToSpread)
{
    const EspParameters esp = makeDerivativeEsp();
    const std::array<real, DIM> x = { real(0.22), real(0.41), real(0.68) };
    const std::array<real, DIM> dx = { real(1e-3), real(0), real(0) };
    const std::vector<real> gridWeights = { real(0.7), real(-0.2), real(0.4), real(0.9), real(-0.1) };

    auto weightedSpread = [&](const std::array<real, DIM>& xEval)
    {
        const std::vector<real> rho = evaluateSpread(esp, xEval);
        real value = real(0);
        for (int k = 0; k < esp.P; ++k)
        {
            value += rho[XX * esp.P_padded + k] * gridWeights[k];
        }
        return value;
    };

    const std::vector<real> drho = evaluateDerivative(esp, x);
    real gatherDerivative = real(0);
    for (int k = 0; k < esp.P; ++k)
    {
        gatherDerivative += drho[XX * esp.P_padded + k] * gridWeights[k];
    }

    std::array<real, DIM> xPlus  = x;
    std::array<real, DIM> xMinus = x;
    xPlus[XX] += dx[XX];
    xMinus[XX] -= dx[XX];
    const real finiteDifference = (weightedSpread(xPlus) - weightedSpread(xMinus)) / (real(2) * dx[XX]);

    EXPECT_NEAR(gatherDerivative, finiteDifference, real(1e-4));
}

TEST(EspGather, AnalyticDerivativeMatchesFiniteDiff)
{
    const EspParameters esp = makeDerivativeEsp();
    const std::array<real, DIM> x = { real(0.17), real(0.53), real(0.81) };
    const std::vector<real> drho = evaluateDerivative(esp, x);

    for (int dim = 0; dim < DIM; ++dim)
    {
        for (int k = 0; k < esp.P_padded; ++k)
        {
            const real expected =
                    scalarHorner(gmx::makeConstArrayRef(esp.drho_coeff), esp.poly_order - 1, esp.P_padded, x[dim], k);
            EXPECT_NEAR(drho[dim * esp.P_padded + k], expected, real(1e-6))
                    << "dim=" << dim << " k=" << k;
        }
    }
}

} // namespace
} // namespace gmx::test
