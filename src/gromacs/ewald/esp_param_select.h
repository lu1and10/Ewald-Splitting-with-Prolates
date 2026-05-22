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

#ifndef GMX_EWALD_ESP_PARAM_SELECT_H
#define GMX_EWALD_ESP_PARAM_SELECT_H

#include <cstdint>

#include "gromacs/mdtypes/inputrec.h"
#include "gromacs/utility/real.h"
#include "gromacs/utility/vectypes.h"

namespace gmx
{
class MDLogger;
}

namespace gmx::esp
{

struct EspAutotuneInput
{
    //! Target relative force accuracy epsilon.
    real accuracy = 0;
    //! Target spreading-window accuracy epsilon_1.
    real spreadAccuracy = 0;
    //! Direct-space cutoff r_c.
    real cutoff = 0;
    //! Simulation box.
    matrix box = { { 0 } };
    //! Number of atoms.
    int64_t natoms = 0;
    //! Sum of squared charges.
    double q2sum = 0;
    //! User stencil-order override. -1 means automatic selection.
    int stencilOrderOverride = -1;
};

//! Run ESP parameter autotuning from validated inputs.
EspParameters autotuneEsp(const EspAutotuneInput& in, const gmx::MDLogger& mdlog);

} // namespace gmx::esp

#endif // GMX_EWALD_ESP_PARAM_SELECT_H
