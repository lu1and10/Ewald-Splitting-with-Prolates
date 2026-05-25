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
 *
 * To help us fund GROMACS development, we humbly ask that you cite
 * the research papers on the package. Check out https://www.gromacs.org.
 */
/*! \internal \file
 * \brief
 * Tests for ESP mdp parsing and check_ir validation.
 *
 * \ingroup module_gmxpreprocess
 */

#include "gmxpre.h"

#include <string>

#include <gtest/gtest.h>

#include "gromacs/fileio/warninp.h"
#include "gromacs/gmxpreprocess/readir.h"
#include "gromacs/mdrun/mdmodules.h"
#include "gromacs/mdtypes/inputrec.h"
#include "gromacs/utility/smalloc.h"
#include "gromacs/utility/stringutil.h"
#include "gromacs/utility/textwriter.h"

#include "testutils/testasserts.h"
#include "testutils/testfilemanager.h"

namespace gmx
{
namespace test
{
namespace
{

std::string makeEspMdp(const char* pbc           = "xyz",
                       const char* freeEnergy    = "no",
                       const char* pressureCoupl = "no",
                       int         nwall         = 0,
                       const char* ewaldGeometry = "3d",
                       const char* accuracy      = "1e-4",
                       const char* rcoulomb      = "1.0",
                       const char* rvdw          = "1.0")
{
    std::string mdp = formatString(
            "integrator = md\n"
            "nsteps = 0\n"
            "dt = 0.001\n"
            "coulombtype = ESP\n"
            "rcoulomb = %s\n"
            "rvdw = %s\n"
            "rlist = 1.0\n"
            "vdwtype = Cut-off\n"
            "pbc = %s\n"
            "free-energy = %s\n"
            "pcoupl = %s\n"
            "ewald-geometry = %s\n"
            "nwall = %d\n"
            "esp-accuracy = %s\n"
            "esp-stencil-order = 12\n"
            "pme-order = 16\n",
            rcoulomb,
            rvdw,
            pbc,
            freeEnergy,
            pressureCoupl,
            ewaldGeometry,
            nwall,
            accuracy);
    if (nwall > 0)
    {
        mdp += "wall-type = 12-6\n"
               "wall-atomtype = OW\n";
    }
    return mdp;
}

class EspMdpValidationTest : public ::testing::Test
{
public:
    EspMdpValidationTest()
    {
        snew(opts_.include, STRLEN);
        snew(opts_.define, STRLEN);
    }

    ~EspMdpValidationTest() override
    {
        done_inputrec_strings();
        sfree(opts_.include);
        sfree(opts_.define);
    }

    bool parseAndCheckMdp(const std::string& mdpContents)
    {
        WarningHandler    wi{ false, 0 };
        const std::string mdpFilename = fileManager_.getTemporaryFilePath("esp.mdp").string();
        TextWriter::writeFileFromString(mdpFilename, mdpContents);

        get_ir(mdpFilename.c_str(), nullptr, &mdModules_, &ir_, &opts_, WriteMdpHeader::no, &wi);
        check_ir(mdpFilename.c_str(), &mdModules_, &ir_, &opts_, &wi);

        return warning_errors_exist(wi);
    }

protected:
    TestFileManager fileManager_;
    t_inputrec      ir_;
    MDModules       mdModules_;
    t_gromppopts    opts_;
};

TEST_F(EspMdpValidationTest, AcceptsValidMdpAndSetsAutoSpreadAccuracy)
{
    EXPECT_FALSE(parseAndCheckMdp(makeEspMdp()));
    EXPECT_EQ(ir_.coulombtype, CoulombInteractionType::Esp);
    EXPECT_REAL_EQ(ir_.espSettings.accuracy, 1e-4_real);
    EXPECT_REAL_EQ(ir_.espSettings.spreadAccuracy, 5e-5_real);
    EXPECT_EQ(ir_.espSettings.stencilOrder, 12);
    EXPECT_EQ(ir_.pme_order, 16);
}

TEST_F(EspMdpValidationTest, ErrorsOnAccuracyOutOfRange)
{
    EXPECT_TRUE(parseAndCheckMdp(makeEspMdp("xyz", "no", "no", 0, "3d", "1e-8")));
}

TEST_F(EspMdpValidationTest, FatalsOnNonXyzPbc)
{
    GMX_EXPECT_DEATH_IF_SUPPORTED(parseAndCheckMdp(makeEspMdp("xy")),
                                  "ESP only supports 3D periodic boundary");
}

TEST_F(EspMdpValidationTest, FatalsOnFep)
{
    GMX_EXPECT_DEATH_IF_SUPPORTED(parseAndCheckMdp(makeEspMdp("xyz", "yes")),
                                  "ESP is not yet compatible with free-energy");
}

TEST_F(EspMdpValidationTest, FatalsOnNpt)
{
    GMX_EXPECT_DEATH_IF_SUPPORTED(parseAndCheckMdp(makeEspMdp("xyz", "no", "berendsen")),
                                  "ESP does not yet implement pressure-tensor");
}

TEST_F(EspMdpValidationTest, FatalsOnNonThreeDEwaldGeometry)
{
    GMX_EXPECT_DEATH_IF_SUPPORTED(parseAndCheckMdp(makeEspMdp("xyz", "no", "no", 0, "3dc")),
                                  "ESP only supports 3D Ewald geometry");
}

TEST_F(EspMdpValidationTest, FatalsOnWalls)
{
    GMX_EXPECT_DEATH_IF_SUPPORTED(parseAndCheckMdp(makeEspMdp("xyz", "no", "no", 1)),
                                  "ESP is not compatible with walls");
}

TEST_F(EspMdpValidationTest, ErrorsOnDifferentCoulombAndVdwCutoffs)
{
    EXPECT_TRUE(parseAndCheckMdp(makeEspMdp("xyz", "no", "no", 0, "3d", "1e-4", "1.0", "0.9")));
}

} // namespace
} // namespace test
} // namespace gmx
