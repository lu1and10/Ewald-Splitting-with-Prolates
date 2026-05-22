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

#include "gmxpre.h"

#include "gromacs/math/pswf.h"

#include <stdexcept>
#include <string>

#include <gtest/gtest.h>
#include <tinyxml2.h>

#include "testutils/testfilemanager.h"

namespace gmx::esp::test
{
namespace
{

TEST(Pswf0, ConstructionRangeChecks)
{
    EXPECT_THROW(Pswf0(0.0), std::invalid_argument);
    EXPECT_THROW(Pswf0(-1.0), std::invalid_argument);
    EXPECT_THROW(Pswf0(30.01), std::invalid_argument);
    EXPECT_NO_THROW(Pswf0(5.0));
    EXPECT_NO_THROW(Pswf0(12.024));
}

TEST(Pswf0, EvalAtZeroEqualsOneAfterNormalization)
{
    Pswf0 psi(5.0);
    EXPECT_NEAR(psi.eval(0.0), 1.0, 1e-12);
}

TEST(Pswf0, EvalSymmetric)
{
    Pswf0 psi(8.0);
    EXPECT_NEAR(psi.eval(0.3), psi.eval(-0.3), 1e-12);
    EXPECT_NEAR(psi.eval(0.7), psi.eval(-0.7), 1e-12);
}

TEST(Pswf0, EvalReturnsZeroOutsideSupport)
{
    Pswf0 psi(5.0);
    EXPECT_DOUBLE_EQ(psi.eval(1.5), 0.0);
    EXPECT_DOUBLE_EQ(psi.eval(-2.0), 0.0);
}

TEST(Pswf0, MatchesMpmathRefdataAt50Digits)
{
    tinyxml2::XMLDocument doc;
    const auto refPath = gmx::test::TestFileManager::getInputFilePath("refdata/pswf_reference.xml");
    ASSERT_EQ(doc.LoadFile(refPath.string().c_str()), tinyxml2::XML_SUCCESS);

    const auto* root = doc.FirstChildElement("PswfRefdata");
    ASSERT_NE(root, nullptr);

    for (const auto* cNode = root->FirstChildElement("CValue"); cNode != nullptr;
         cNode             = cNode->NextSiblingElement("CValue"))
    {
        double c = 0.0;
        ASSERT_EQ(cNode->QueryDoubleAttribute("c", &c), tinyxml2::XML_SUCCESS);

        Pswf0 psi(c);
        for (const auto* sample = cNode->FirstChildElement("Sample"); sample != nullptr;
             sample            = sample->NextSiblingElement("Sample"))
        {
            double x = 0.0;
            ASSERT_EQ(sample->QueryDoubleAttribute("x", &x), tinyxml2::XML_SUCCESS);
            ASSERT_NE(sample->GetText(), nullptr);

            const double referenceValue = std::stod(sample->GetText());
            EXPECT_NEAR(psi.eval(x), referenceValue, 1e-10) << "c=" << c << " x=" << x;
        }
    }
}

TEST(Pswf0, EvalDerivativeVsFiniteDifference)
{
    Pswf0        psi(8.0);
    const double h = 1e-6;
    for (double x : { -0.7, -0.3, 0.0, 0.3, 0.7 })
    {
        const double finiteDifference = (psi.eval(x + h) - psi.eval(x - h)) / (2.0 * h);
        EXPECT_NEAR(psi.evalDerivative(x), finiteDifference, 1e-7) << "x=" << x;
    }
}

TEST(Pswf0, EvalDerivativeAtZeroIsZero)
{
    Pswf0 psi(8.0);
    EXPECT_NEAR(psi.evalDerivative(0.0), 0.0, 1e-12);
}

} // namespace
} // namespace gmx::esp::test
