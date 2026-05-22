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

#include "gromacs/ewald/esp_param_select.h"

#include <gtest/gtest.h>

#include "gromacs/math/pswf.h"
#include "gromacs/utility/logger.h"

namespace gmx::esp::test
{
namespace
{

constexpr real c_pi = 3.14159265358979323846_real;

const gmx::MDLogger nullLogger;

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

} // namespace
} // namespace gmx::esp::test
