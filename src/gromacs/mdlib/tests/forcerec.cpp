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

#include "gromacs/mdlib/forcerec.h"

#include <memory>

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

TEST(Forcerec, EspInitializesEspShortRangeTable)
{
    interaction_const_t ic;
    ic.coulomb.type       = CoulombInteractionType::Esp;
    ic.coulomb.cutoff     = 2.0_real;
    ic.coulomb.ewaldCoeff = 0.5_real;
    ic.coulombEwaldTables = std::make_unique<EwaldCorrectionTables>();
    ic.vdw.type           = VanDerWaalsType::Cut;
    ic.vdwEwaldTables     = std::make_unique<EwaldCorrectionTables>();

    ic.esp.cutoff            = 2.0_real;
    ic.esp.relativeTolerance = 1e-4_real;
    ic.esp.splitCoefficient  = static_cast<real>(gmx::esp::prolc180(ic.esp.relativeTolerance));

    const real tableScale = ewald_spline3_table_scale(ic, true, false);
    const int  tableSize  = static_cast<int>(ic.coulomb.cutoff * tableScale) + 2;

    const EwaldCorrectionTables expectedTable = generateEspShortRangeTable(ic, tableSize, tableScale);

    ::init_interaction_const_tables(nullptr, &ic, ic.coulomb.cutoff, 0.0_real);

    const EwaldCorrectionTables& table = *ic.coulombEwaldTables;
    ASSERT_GE(table.tableV.size(), 2U);
    ASSERT_EQ(table.tableF.size(), table.tableV.size());
    ASSERT_EQ(table.tableFDV0.size(), 4U * table.tableV.size());
    EXPECT_REAL_EQ(table.scale, expectedTable.scale);

    for (const int i :
         { 0, static_cast<int>(table.tableV.size() / 2), static_cast<int>(table.tableV.size() - 1) })
    {
        EXPECT_REAL_EQ(table.tableV[i], expectedTable.tableV[i]);
        EXPECT_REAL_EQ(table.tableF[i], expectedTable.tableF[i]);
    }
}

} // namespace
} // namespace gmx::test
