/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright 2019- The GROMACS Authors
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
/*!\internal
 * \file
 * \brief
 * Implements gmx::OutputAdapterContainer.
 *
 * \author Paul Bauer <paul.bauer.q@gmail.com>
 * \ingroup module_coordinateio
 */

#include "gmxpre.h"

#include "outputadaptercontainer.h"

#include <algorithm>
#include <utility>

#include "gromacs/utility/exceptions.h"

namespace gmx
{
enum class CoordinateFileFlags : unsigned long;

void OutputAdapterContainer::addAdapter(OutputAdapterPointer adapter, CoordinateFileFlags type)
{
    if (outputAdapters_[type] != nullptr)
    {
        GMX_THROW(InternalError("Trying to add adapter that has already been added"));
    }
    adapter->checkAbilityDependencies(abilities_);
    outputAdapters_[type] = std::move(adapter);
}

bool OutputAdapterContainer::isEmpty() const
{
    return std::none_of(outputAdapters_.begin(),
                        outputAdapters_.end(),
                        [](const auto& adapter) { return adapter != nullptr; });
}
} // namespace gmx
