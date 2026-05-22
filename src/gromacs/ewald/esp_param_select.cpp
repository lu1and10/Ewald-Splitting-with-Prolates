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

#include <cmath>

#include "gromacs/fft/calcgrid.h"
#include "gromacs/math/pswf.h"
#include "gromacs/simd/simd.h"
#include "gromacs/utility/gmxassert.h"
#include "gromacs/utility/logger.h"

namespace gmx::esp
{

EspParameters autotuneEsp(const EspAutotuneInput& in, const gmx::MDLogger& /*mdlog*/)
{
    GMX_ASSERT(in.accuracy > 0, "ESP autotune: accuracy validated upstream");
    GMX_ASSERT(in.cutoff > 0, "ESP autotune: cutoff validated upstream");
    GMX_ASSERT(in.q2sum > 0, "ESP autotune: q2sum validated upstream");
    GMX_ASSERT(GMX_SIMD_REAL_WIDTH > 0, "SIMD width must be positive");

    EspParameters out;

    out.c  = static_cast<real>(prolc180(static_cast<double>(in.accuracy)));
    out.c1 = static_cast<real>(prolc180(0.5 * static_cast<double>(in.spreadAccuracy)));
    out.P  = (in.stencilOrderOverride > 0)
                     ? in.stencilOrderOverride
                     : estimateOrder(static_cast<double>(in.accuracy));
    out.P_padded = ((out.P + GMX_SIMD_REAL_WIDTH - 1) / GMX_SIMD_REAL_WIDTH) * GMX_SIMD_REAL_WIDTH;

    const real h0 = static_cast<real>(M_PI) * in.cutoff / out.c;
    calcFftGrid(nullptr, in.box, h0, 2, &out.nx, &out.ny, &out.nz);

    const Pswf0 pswfC(out.c);
    const Pswf0 pswfC1(out.c1);
    out.lambda0    = static_cast<real>(pswfC.lambda0());
    out.psi0AtZero = static_cast<real>(pswfC.eval(0.0));
    out.lambda0_w  = static_cast<real>(pswfC1.lambda0());
    out.selfCoeff  = -1.0_real / (in.cutoff * out.lambda0);
    out.cutoff     = in.cutoff;

    spreadRealPoly(out.P,
                   out.P_padded,
                   static_cast<double>(in.spreadAccuracy),
                   static_cast<double>(in.spreadAccuracy) * 0.1,
                   static_cast<double>(out.c1),
                   &out.rho_coeff,
                   &out.poly_order);

    out.drho_coeff.assign(static_cast<size_t>(out.poly_order) * out.P_padded, 0.0_real);
    for (int l = 0; l < out.poly_order - 1; ++l)
    {
        const real scale = static_cast<real>(l + 1);
        for (int k = 0; k < out.P_padded; ++k)
        {
            out.drho_coeff[l * out.P_padded + k] =
                    scale * out.rho_coeff[(l + 1) * out.P_padded + k];
        }
    }

    spreadFourierPoly(static_cast<double>(in.spreadAccuracy),
                      static_cast<double>(in.spreadAccuracy) * 0.1,
                      static_cast<double>(out.c1),
                      &out.spread_fourier_poly,
                      &out.spread_fourier_poly_order);

    splitFourierPoly(static_cast<double>(in.accuracy),
                     static_cast<double>(in.accuracy) * 0.1,
                     static_cast<double>(out.c),
                     &out.split_fourier_poly,
                     &out.split_fourier_poly_order);

    shortRangeForcePoly(static_cast<double>(in.accuracy),
                        static_cast<double>(in.accuracy) * 0.1,
                        static_cast<double>(out.c),
                        &out.short_range_force_poly,
                        &out.short_range_force_poly_order);

    shortRangeEnergyPoly(static_cast<double>(in.accuracy) * 0.01,
                         static_cast<double>(in.accuracy) * 0.001,
                         static_cast<double>(out.c),
                         &out.short_range_energy_poly,
                         &out.short_range_energy_poly_order);

    return out;
}

} // namespace gmx::esp
