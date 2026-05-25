/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright 2016- The GROMACS Authors
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

#include <cmath>

#include <gtest/gtest.h>

#include "gromacs/math/pswf.h"
#include "gromacs/mdtypes/interaction_const.h"
#include "gromacs/mdtypes/md_enums.h"
#include "gromacs/tables/forcetable.h"
#include "gromacs/utility/real.h"

#include "testutils/testasserts.h"

namespace gmx::test
{
namespace
{

double exactEspTablePotential(const gmx::esp::Pswf0& psi, const double cutoff, const double r)
{
    if (r == 0.0)
    {
        return 2.0 * psi.eval(0.0) / (psi.lambda0() * cutoff);
    }

    return gmx::esp::pswfSplitFunction(psi, 1.0 / cutoff, r) / r;
}

interaction_const_t makeEspTableInteractionConst(const real ewaldCoeff, const real relativeTolerance)
{
    interaction_const_t ic;
    ic.coulomb.type          = CoulombInteractionType::Esp;
    ic.coulomb.cutoff        = 2.0_real;
    ic.coulomb.ewaldCoeff    = ewaldCoeff;
    ic.vdw.type              = VanDerWaalsType::Cut;
    ic.esp.cutoff            = ic.coulomb.cutoff;
    ic.esp.relativeTolerance = relativeTolerance;
    ic.esp.splitCoefficient =
            static_cast<real>(gmx::esp::prolc180(static_cast<double>(relativeTolerance)));
    return ic;
}

TEST(EspShortRangeTable, SamplesExactPswfSplitFunction)
{
    interaction_const_t ic;
    ic.esp.cutoff           = 2.0_real;
    ic.esp.splitCoefficient = 6.0_real;

    const EwaldCorrectionTables table = generateEspShortRangeTable(ic, 65);
    const gmx::esp::Pswf0       psi(ic.esp.splitCoefficient);

    ASSERT_EQ(table.tableV.size(), 65U);
    ASSERT_EQ(table.tableF.size(), 65U);
    ASSERT_EQ(table.tableFDV0.size(), 65U * 4U);
    EXPECT_NEAR(table.scale, 32.0_real, 1e-6_real);

    for (int i : { 0, 16, 32, 48, 64 })
    {
        const real r = i / table.scale;
        EXPECT_NEAR(table.tableV[i], exactEspTablePotential(psi, ic.esp.cutoff, r), 1e-6_real);
    }

    for (int i = 0; i < 64; ++i)
    {
        EXPECT_REAL_EQ(table.tableFDV0[4 * i], table.tableF[i]);
        EXPECT_REAL_EQ(table.tableFDV0[4 * i + 1], table.tableF[i + 1] - table.tableF[i]);
        EXPECT_REAL_EQ(table.tableFDV0[4 * i + 2], table.tableV[i]);
    }
}

TEST(EspShortRangeTable, SplineScaleDoesNotUsePmeEwaldCoeff)
{
    interaction_const_t lowPmeEwaldCoeff  = makeEspTableInteractionConst(0.5_real, 1e-4_real);
    interaction_const_t highPmeEwaldCoeff = makeEspTableInteractionConst(5.0_real, 1e-4_real);

    const real lowScale  = ewald_spline3_table_scale(lowPmeEwaldCoeff, true, false);
    const real highScale = ewald_spline3_table_scale(highPmeEwaldCoeff, true, false);

    EXPECT_REAL_EQ(lowScale, highScale);
}

TEST(EspShortRangeTable, SplineScaleTightensWithEspAccuracy)
{
    interaction_const_t loose = makeEspTableInteractionConst(0.5_real, 1e-2_real);
    interaction_const_t tight = makeEspTableInteractionConst(0.5_real, 1e-6_real);

    const real looseScale = ewald_spline3_table_scale(loose, true, false);
    const real tightScale = ewald_spline3_table_scale(tight, true, false);

    EXPECT_GT(tightScale, looseScale);
}

} // namespace
} // namespace gmx::test
