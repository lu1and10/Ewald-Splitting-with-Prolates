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
 */
/*! \internal \file
 * \brief
 * Tests for tpxio.
 *
 * \ingroup module_fileio
 */

#include "gmxpre.h"

#include <filesystem>
#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "gromacs/fileio/tpxio.h"
#include "gromacs/mdtypes/inputrec.h"
#include "gromacs/mdtypes/state.h"
#include "gromacs/topology/symtab.h"
#include "gromacs/topology/topology.h"
#include "gromacs/utility/keyvaluetree.h"
#include "gromacs/utility/real.h"

#include "testutils/testasserts.h"
#include "testutils/testfilemanager.h"

namespace gmx
{
namespace test
{
namespace
{

void initializeRequiredInputrecTrees(t_inputrec* ir)
{
    ir->params             = new KeyValueTreeObject();
    ir->internalParameters = std::make_unique<KeyValueTreeObject>();
}

TEST(TpxIoEsp, RoundTripsEspParameters)
{
    TestFileManager fileManager;
    const std::filesystem::path tprFilename = fileManager.getTemporaryFilePath("esp-roundtrip.tpr");

    t_inputrec ir;
    initializeRequiredInputrecTrees(&ir);
    ir.eI                         = IntegrationAlgorithm::MD;
    ir.pbcType                    = PbcType::Xyz;
    ir.ensembleTemperatureSetting = EnsembleTemperatureSetting::NotAvailable;
    ir.ensembleTemperature        = -1;
    ir.coulombtype                = CoulombInteractionType::Esp;
    ir.rcoulomb                   = 1.2_real;
    ir.fepvals->sc_r_power        = 6.0_real;
    ir.espSettings.accuracy       = 1e-4_real;
    ir.espSettings.spreadAccuracy = 2.5e-5_real;
    ir.espSettings.stencilOrder   = 7;

    ir.espParams.c                       = 12.024_real;
    ir.espParams.lambda0                 = 0.8125_real;
    ir.espParams.psi0AtZero              = 1.0_real;
    ir.espParams.selfCoeff               = -1.025641025641_real;
    ir.espParams.cutoff                  = ir.rcoulomb;
    ir.espParams.c1                      = 9.125_real;
    ir.espParams.lambda0_w               = 0.734375_real;
    ir.espParams.P                       = 7;
    ir.espParams.P_padded                = 8;
    ir.espParams.nx                      = 24;
    ir.espParams.ny                      = 30;
    ir.espParams.nz                      = 36;
    ir.espParams.poly_order              = 4;
    ir.espParams.rho_coeff               = { 0.125_real, 0.25_real, 0.5_real, 1.0_real };
    ir.espParams.drho_coeff              = { -0.375_real, -0.25_real, -0.125_real, 0.0_real };
    ir.espParams.split_fourier_poly_order = 3;
    ir.espParams.split_fourier_poly       = { 1.0_real, -0.5_real, 0.25_real };
    ir.espParams.spread_fourier_poly_order = 3;
    ir.espParams.spread_fourier_poly       = { 1.0_real, -0.25_real, 0.0625_real };
    ir.espParams.short_range_force_poly_order = 4;
    ir.espParams.short_range_force_poly = {
        0.875_real, -0.125_real, 0.03125_real, -0.0078125_real
    };
    ir.espParams.short_range_energy_poly_order = 4;
    ir.espParams.short_range_energy_poly = { -0.75_real, 0.25_real, -0.0625_real, 0.015625_real };

    t_state state;
    state.box[XX][XX] = 2.5_real;
    state.box[YY][YY] = 3.0_real;
    state.box[ZZ][ZZ] = 3.5_real;
    gmx_mtop_t mtop;
    mtop.name = put_symtab(&mtop.symtab, "ESP round-trip test");

    write_tpx_state(tprFilename, &ir, &state, mtop);

    t_inputrec irRead;
    t_state    stateRead;
    gmx_mtop_t mtopRead;
    read_tpx_state(tprFilename, &irRead, &stateRead, &mtopRead);

    EXPECT_EQ(irRead.coulombtype, ir.coulombtype);
    EXPECT_EQ(irRead.espSettings.stencilOrder, ir.espSettings.stencilOrder);
    EXPECT_EQ(irRead.espParams.P, ir.espParams.P);
    EXPECT_EQ(irRead.espParams.P_padded, ir.espParams.P_padded);
    EXPECT_EQ(irRead.espParams.nx, ir.espParams.nx);
    EXPECT_EQ(irRead.espParams.ny, ir.espParams.ny);
    EXPECT_EQ(irRead.espParams.nz, ir.espParams.nz);
    EXPECT_REAL_EQ(irRead.espSettings.accuracy, ir.espSettings.accuracy);
    EXPECT_REAL_EQ(irRead.espSettings.spreadAccuracy, ir.espSettings.spreadAccuracy);
    EXPECT_REAL_EQ(irRead.espParams.c, ir.espParams.c);
    EXPECT_REAL_EQ(irRead.espParams.lambda0, ir.espParams.lambda0);
    EXPECT_REAL_EQ(irRead.espParams.selfCoeff, ir.espParams.selfCoeff);
    EXPECT_REAL_EQ(irRead.espParams.c1, ir.espParams.c1);
    EXPECT_THAT(irRead.espParams.rho_coeff, ::testing::ElementsAreArray(ir.espParams.rho_coeff));
    EXPECT_THAT(irRead.espParams.drho_coeff, ::testing::ElementsAreArray(ir.espParams.drho_coeff));
    EXPECT_THAT(irRead.espParams.split_fourier_poly,
                ::testing::ElementsAreArray(ir.espParams.split_fourier_poly));
    EXPECT_THAT(irRead.espParams.spread_fourier_poly,
                ::testing::ElementsAreArray(ir.espParams.spread_fourier_poly));
    EXPECT_THAT(irRead.espParams.short_range_force_poly,
                ::testing::ElementsAreArray(ir.espParams.short_range_force_poly));
    EXPECT_THAT(irRead.espParams.short_range_energy_poly,
                ::testing::ElementsAreArray(ir.espParams.short_range_energy_poly));
}

} // namespace
} // namespace test
} // namespace gmx
