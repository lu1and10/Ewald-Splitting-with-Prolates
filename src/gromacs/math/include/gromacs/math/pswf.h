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

#ifndef GMX_MATH_PSWF_H
#define GMX_MATH_PSWF_H

#include <vector>

#include "gromacs/utility/alignedallocator.h"
#include "gromacs/utility/real.h"

namespace gmx::esp
{

//! Vector type alias for SIMD-aligned real arrays used by ESP polynomial tables.
//! Avoids dragging in gpu_utils/hostallocator.h transitively.
using AlignedRealVector = std::vector<real, gmx::AlignedAllocator<real>>;

/*! \brief First-order prolate spheroidal wave function psi_0^c on [-1, 1].
 *
 * Constructed from a bandlimit parameter c via a Legendre expansion. Outside
 * [-1, 1] eval() returns zero.
 */
class Pswf0
{
public:
    //! Construct psi_0^c. Throws std::invalid_argument if c <= 0 or c > 30.
    explicit Pswf0(double c);

    //! Bandlimit parameter c.
    double c() const noexcept { return c_; }

    //! Largest eigenvalue lambda_0 of the integral operator F_c.
    double lambda0() const noexcept { return lambda0_; }

    //! Value of psi_0^c at x in [-1, 1]. Returns zero outside support.
    double eval(double x) const;

    //! Derivative psi_0^c'(x) on [-1, 1].
    double evalDerivative(double x) const;

    //! Integral int_0^upper psi_0^c(u) du for upper in [-1, 1].
    double evalIntegral(double upper) const;

private:
    double              c_;
    double              lambda0_;
    double              normalizationAt0_;
    std::vector<double> legendreCoefficients_;
};

//! Tolerance to PSWF bandlimit c.
double prolc180(double tolerance);

//! d c / d epsilon for downstream chain rule.
double prolc180Der(double tolerance);

//! Compactly-supported PSWF split function Phi_{r_c}^{c_s}(x) at scalar x >= 0.
double pswfSplitFunction(const Pswf0& psi, double rcInv, double x);

//! epsilon to stencil order P, clamped to [4, 16].
int estimateOrder(double tolerance);

//! Real-space spreading window polynomial table, lane-major in the inner P axis.
void spreadRealPoly(int                P,
                    int                P_padded,
                    double             tol,
                    double             r_tol,
                    double             c_w,
                    AlignedRealVector* coefs,
                    int*               polyOrderOut);

//! Fourier-space normalized |phihat_1D|^2(s)/|phihat_1D|^2(0) on s in [0, 1].
void spreadFourierPoly(double             tol,
                       double             r_tol,
                       double             c_w,
                       AlignedRealVector* coefs,
                       int*               polyOrderOut);

//! Short-range force kernel d/dr [(1 - Phi_{r_c}(r)) / r] on [0, r_c].
void shortRangeForcePoly(double tol, double r_tol, double c, AlignedRealVector* coefs, int* polyOrderOut);

//! Short-range energy kernel (1 - Phi_{r_c}(r)) / r on [0, r_c].
void shortRangeEnergyPoly(double tol, double r_tol, double c, AlignedRealVector* coefs, int* polyOrderOut);

//! Fourier-space chihat(arg), where arg = r_c * |q|.
void splitFourierPoly(double tol, double r_tol, double c, AlignedRealVector* coefs, int* polyOrderOut);

} // namespace gmx::esp

#endif // GMX_MATH_PSWF_H
