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

/* The Pswf0 Legendre expansion implementation in later commits is derived
 * from FINUFFT src/common/pswf.cpp, Apache-2.0 licensed, with user approval.
 */

#include "gmxpre.h"

#include "gromacs/math/pswf.h"

#include <stdexcept>

namespace gmx::esp
{

Pswf0::Pswf0(double c) : c_(c), lambda0_(0.0), normalizationAt0_(1.0)
{
    if (c <= 0.0 || c > 30.0)
    {
        throw std::invalid_argument("Pswf0: c must be in (0, 30]");
    }
}

double Pswf0::eval(double /*x*/) const
{
    return 0.0;
}

double Pswf0::evalDerivative(double /*x*/) const
{
    return 0.0;
}

double Pswf0::evalIntegral(double /*upper*/) const
{
    return 0.0;
}

double prolc180(double /*tolerance*/)
{
    return 0.0;
}

double prolc180Der(double /*tolerance*/)
{
    return 0.0;
}

double pswfSplitFunction(const Pswf0& /*psi*/, double /*rcInv*/, double /*x*/)
{
    return 0.0;
}

int estimateOrder(double /*tolerance*/)
{
    return 4;
}

void spreadRealPoly(int /*P*/,
                    int /*P_padded*/,
                    double /*tol*/,
                    double /*r_tol*/,
                    double /*c_w*/,
                    AlignedRealVector* coefs,
                    int*               polyOrderOut)
{
    coefs->clear();
    *polyOrderOut = 0;
}

void spreadFourierPoly(double /*tol*/,
                       double /*r_tol*/,
                       double /*c_w*/,
                       AlignedRealVector* coefs,
                       int*               polyOrderOut)
{
    coefs->clear();
    *polyOrderOut = 0;
}

void shortRangeForcePoly(double, double, double, AlignedRealVector* coefs, int* polyOrderOut)
{
    coefs->clear();
    *polyOrderOut = 0;
}

void shortRangeEnergyPoly(double, double, double, AlignedRealVector* coefs, int* polyOrderOut)
{
    coefs->clear();
    *polyOrderOut = 0;
}

void splitFourierPoly(double, double, double, AlignedRealVector* coefs, int* polyOrderOut)
{
    coefs->clear();
    *polyOrderOut = 0;
}

} // namespace gmx::esp
