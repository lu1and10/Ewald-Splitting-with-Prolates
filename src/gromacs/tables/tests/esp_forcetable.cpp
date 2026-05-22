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
 * GROMACS is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public
 * License for more details.
 */

#include "gmxpre.h"

#include "gromacs/tables/forcetable.h"

#include <gtest/gtest.h>

#include "gromacs/mdtypes/interaction_const.h"
#include "gromacs/utility/real.h"

namespace gmx::test
{
namespace
{

real evaluatePolynomial(const gmx::esp::AlignedRealVector& coefs, int order, real s)
{
    real value = coefs[order - 1];
    for (int i = order - 2; i >= 0; --i)
    {
        value = value * s + coefs[i];
    }
    return value;
}

TEST(EspShortRangeTable, MatchesPolynomial)
{
    interaction_const_t ic;
    ic.esp.cutoff          = 2.0_real;
    ic.esp.energyPolyOrder = 3;
    ic.esp.energyPolyCoeff = { 1.0_real, -0.25_real, 0.125_real };
    ic.esp.forcePolyOrder  = 3;
    ic.esp.forcePolyCoeff  = { -1.0_real, 0.5_real, -0.125_real };

    const EwaldCorrectionTables table = generateEspShortRangeTable(ic, 17);

    ASSERT_EQ(table.tableV.size(), 17U);
    ASSERT_EQ(table.tableF.size(), 17U);
    ASSERT_EQ(table.tableFDV0.size(), 17U * 4U);
    EXPECT_NEAR(table.scale, 8.0_real, 1e-6_real);

    for (int i : { 0, 4, 8, 12, 16 })
    {
        const real r = i / table.scale;
        const real s = r / ic.esp.cutoff;
        EXPECT_NEAR(table.tableV[i],
                    evaluatePolynomial(ic.esp.energyPolyCoeff, ic.esp.energyPolyOrder, s)
                            / ic.esp.cutoff,
                    1e-6_real);
        EXPECT_NEAR(table.tableF[i],
                    -s * evaluatePolynomial(ic.esp.forcePolyCoeff, ic.esp.forcePolyOrder, s)
                            / ic.esp.cutoff,
                    1e-6_real);
    }
}

} // namespace
} // namespace gmx::test
