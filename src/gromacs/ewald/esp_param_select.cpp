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

#include <algorithm>
#include <exception>

#include "gromacs/fft/calcgrid.h"
#include "gromacs/math/pswf.h"
#include "gromacs/simd/simd.h"
#include "gromacs/utility/fatalerror.h"
#include "gromacs/utility/gmxassert.h"
#include "gromacs/utility/logger.h"

namespace gmx::esp
{
namespace
{

constexpr int  c_minEspStencilOrder = 4;
constexpr int  c_maxEspStencilOrder = 12;
constexpr int  c_maxEspGridSize     = 4096;
constexpr real c_minEspAccuracy     = 1e-7_real;
constexpr real c_maxEspAccuracy     = 1e-2_real;

void checkAutotuneInput(const EspAutotuneInput& in)
{
    if (!std::isfinite(in.accuracy) || in.accuracy < c_minEspAccuracy || in.accuracy > c_maxEspAccuracy)
    {
        gmx_fatal(FARGS,
                  "ESP autotune accuracy must be finite and in supported range [%g, %g], got %g",
                  static_cast<double>(c_minEspAccuracy),
                  static_cast<double>(c_maxEspAccuracy),
                  static_cast<double>(in.accuracy));
    }
    if (!std::isfinite(in.spreadAccuracy) || in.spreadAccuracy <= 0.0_real || in.spreadAccuracy >= 1.0_real)
    {
        gmx_fatal(FARGS,
                  "ESP autotune spread accuracy must be finite and in (0, 1), got %g",
                  static_cast<double>(in.spreadAccuracy));
    }
    if (!std::isfinite(in.cutoff) || in.cutoff <= 0.0_real)
    {
        gmx_fatal(FARGS,
                  "ESP autotune cutoff must be finite and positive, got %g",
                  static_cast<double>(in.cutoff));
    }
    if (!std::isfinite(in.q2sum) || in.q2sum <= 0.0)
    {
        gmx_fatal(FARGS, "ESP autotune q2sum must be finite and positive, got %g", in.q2sum);
    }
    if (in.stencilOrderOverride > 0
        && (in.stencilOrderOverride < c_minEspStencilOrder || in.stencilOrderOverride > c_maxEspStencilOrder))
    {
        gmx_fatal(FARGS,
                  "ESP stencil order must be in [%d, %d] if explicitly set, got %d",
                  c_minEspStencilOrder,
                  c_maxEspStencilOrder,
                  in.stencilOrderOverride);
    }
}

double checkedProlc180(double tolerance, const char* name)
{
    try
    {
        return prolc180(tolerance);
    }
    catch (const std::exception& e)
    {
        gmx_fatal(FARGS, "ESP autotune failed while selecting %s: %s", name, e.what());
    }
}

} // namespace

EspParameters autotuneEsp(const EspAutotuneInput& in, const gmx::MDLogger& /*mdlog*/)
{
    checkAutotuneInput(in);
    GMX_ASSERT(GMX_SIMD_REAL_WIDTH > 0, "SIMD width must be positive");

    EspParameters out;

    const real splitAccuracy  = in.accuracy;
    const real spreadAccuracy = in.spreadAccuracy;

    out.c  = static_cast<real>(checkedProlc180(static_cast<double>(splitAccuracy), "split c"));
    out.c1 = static_cast<real>(checkedProlc180(static_cast<double>(spreadAccuracy), "spread c"));
    const int estimatedOrder = estimateOrder(static_cast<double>(splitAccuracy));
    out.P = (in.stencilOrderOverride > 0) ? in.stencilOrderOverride : estimatedOrder;
    if (out.P < c_minEspStencilOrder || out.P > c_maxEspStencilOrder)
    {
        gmx_fatal(FARGS,
                  "ESP stencil order must be in [%d, %d], got %d",
                  c_minEspStencilOrder,
                  c_maxEspStencilOrder,
                  out.P);
    }
    out.P_padded = ((out.P + GMX_SIMD_REAL_WIDTH - 1) / GMX_SIMD_REAL_WIDTH) * GMX_SIMD_REAL_WIDTH;

    const real h0          = static_cast<real>(M_PI) * in.cutoff / out.c;
    const int  minGridSize = 2 * (out.P - 1);
    calcFftGrid(nullptr, in.box, h0, minGridSize, &out.nx, &out.ny, &out.nz);
    if (std::max({ out.nx, out.ny, out.nz }) > c_maxEspGridSize)
    {
        gmx_fatal(FARGS,
                  "ESP grid is too large (%d x %d x %d); max supported grid dimension is %d",
                  out.nx,
                  out.ny,
                  out.nz,
                  c_maxEspGridSize);
    }

    const Pswf0 pswfC(out.c);
    const Pswf0 pswfC1(out.c1);
    out.lambda0    = static_cast<real>(pswfC.lambda0());
    out.psi0AtZero = static_cast<real>(pswfC.eval(0.0));
    out.lambda0_w  = static_cast<real>(pswfC1.lambda0());
    out.selfCoeff  = -1.0_real / (in.cutoff * out.lambda0);
    out.netChargeCorrectionCoeff =
            static_cast<real>(pswfNetChargeCorrectionCoeff(pswfC, static_cast<double>(in.cutoff)));
    out.cutoff = in.cutoff;

    spreadRealPoly(out.P,
                   out.P_padded,
                   static_cast<double>(spreadAccuracy),
                   static_cast<double>(spreadAccuracy) * 0.1,
                   static_cast<double>(out.c1),
                   &out.rho_coeff,
                   &out.poly_order);

    spreadRealDerivativePoly(out.P,
                             out.P_padded,
                             static_cast<double>(spreadAccuracy),
                             static_cast<double>(spreadAccuracy) * 0.01,
                             static_cast<double>(out.c1),
                             &out.drho_coeff,
                             &out.drho_poly_order);

    spreadFourierPoly(static_cast<double>(spreadAccuracy),
                      static_cast<double>(spreadAccuracy) * 0.1,
                      static_cast<double>(out.c1),
                      &out.spread_fourier_poly,
                      &out.spread_fourier_poly_order);

    splitFourierPoly(static_cast<double>(splitAccuracy),
                     static_cast<double>(splitAccuracy) * 0.1,
                     static_cast<double>(out.c),
                     &out.split_fourier_poly,
                     &out.split_fourier_poly_order);

    shortRangeForcePoly(static_cast<double>(splitAccuracy),
                        static_cast<double>(splitAccuracy) * 0.1,
                        static_cast<double>(out.c),
                        &out.short_range_force_poly,
                        &out.short_range_force_poly_order);

    shortRangeEnergyPoly(static_cast<double>(splitAccuracy) * 0.01,
                         static_cast<double>(splitAccuracy) * 0.001,
                         static_cast<double>(out.c),
                         &out.short_range_energy_poly,
                         &out.short_range_energy_poly_order);

    return out;
}

} // namespace gmx::esp
