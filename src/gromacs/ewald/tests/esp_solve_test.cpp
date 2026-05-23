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
 * To help us fund GROMACS development, we humbly ask that you cite
 * the research papers on the package. Check out https://www.gromacs.org.
 */

#include "gmxpre.h"

#include <cmath>

#include <vector>

#include <gtest/gtest.h>

#include "gromacs/ewald/pme_solve.h"
#include "gromacs/simd/simd.h"
#include "gromacs/utility/arrayref.h"
#include "gromacs/utility/vectypes.h"

namespace gmx::test
{
namespace
{

struct SolveFixture
{
    int               nx                  = 10;
    int               ny                  = 8;
    int               nz                  = 6;
    ivec              localOffset         = { 0, 0, 0 };
    ivec              localNData          = { nx, ny, nz };
    real              boxX                = real(3.0);
    real              boxY                = real(4.0);
    real              boxZ                = real(5.0);
    real              recipXX             = real(1.0) / boxX;
    real              recipYX             = real(0);
    real              recipYY             = real(1.0) / boxY;
    real              recipZX             = real(0);
    real              recipZY             = real(0);
    real              recipZZ             = real(1.0) / boxZ;
    real              boxVolume           = boxX * boxY * boxZ;
    bool              customReciprocalBox = false;
    real              cutoff              = real(1.0);
    real              bandlimit           = real(100.0);
    int               stencilOrder        = 2;
    std::vector<real> splitFourierPoly    = { real(1.0) };
    std::vector<real> bspX;
    std::vector<real> bspY;
    std::vector<real> bspZ;
    std::vector<real> solver;
    std::vector<real> scratchQSquared;
    std::vector<real> scratchBspX;
    std::vector<real> scratchChi;
    std::vector<real> scratchPk;

    SolveFixture() { resetStorage(); }

    void resetStorage()
    {
        localNData[XX] = nx;
        localNData[YY] = ny;
        localNData[ZZ] = nz;
        bspX.assign(nx, real(1));
        bspY.assign(ny, real(1));
        bspZ.assign(nz, real(1));
        solver.assign(nx * ny * nz, real(-1));
        const int scratchSize = ((localNData[XX] + GMX_SIMD_REAL_WIDTH - 1) / GMX_SIMD_REAL_WIDTH)
                                * GMX_SIMD_REAL_WIDTH;
        scratchQSquared.assign(scratchSize, real(0));
        scratchBspX.assign(scratchSize, real(0));
        scratchChi.assign(scratchSize, real(0));
        scratchPk.assign(scratchSize, real(0));
    }

    void setTriclinicReciprocalBox(real xx, real yx, real yy, real zx, real zy, real zz)
    {
        recipXX             = xx;
        recipYX             = yx;
        recipYY             = yy;
        recipZX             = zx;
        recipZY             = zy;
        recipZZ             = zz;
        boxVolume           = real(1) / (recipXX * recipYY * recipZZ);
        customReciprocalBox = true;
    }

    int index(int kx, int iy, int iz) const
    {
        return (iz * localNData[YY] + iy) * localNData[XX] + (kx - localOffset[XX]);
    }

    void run()
    {
        if (!customReciprocalBox)
        {
            recipXX   = real(1) / boxX;
            recipYX   = real(0);
            recipYY   = real(1) / boxY;
            recipZX   = real(0);
            recipZY   = real(0);
            recipZZ   = real(1) / boxZ;
            boxVolume = boxX * boxY * boxZ;
        }
        calc_exponentials_pswf(nx,
                               ny,
                               nz,
                               (nx + 1) / 2,
                               localOffset,
                               localNData,
                               recipXX,
                               recipYX,
                               recipYY,
                               recipZX,
                               recipZY,
                               recipZZ,
                               boxVolume,
                               cutoff,
                               bandlimit,
                               stencilOrder,
                               gmx::makeConstArrayRef(splitFourierPoly),
                               gmx::ssize(splitFourierPoly),
                               gmx::makeConstArrayRef(bspX),
                               gmx::makeConstArrayRef(bspY),
                               gmx::makeConstArrayRef(bspZ),
                               gmx::makeArrayRef(solver),
                               gmx::makeArrayRef(scratchQSquared),
                               gmx::makeArrayRef(scratchBspX),
                               gmx::makeArrayRef(scratchChi),
                               gmx::makeArrayRef(scratchPk));
    }
};

int signedWaveNumber(int k, int n)
{
    return (k < (n + 1) / 2) ? k : k - n;
}

real angularInfluenceReference(const SolveFixture& f, int kx, int ky, int kz)
{
    const real qx       = real(2.0 * M_PI) * real(signedWaveNumber(kx, f.nx)) / f.boxX;
    const real qy       = real(2.0 * M_PI) * real(signedWaveNumber(ky, f.ny)) / f.boxY;
    const real qz       = real(2.0 * M_PI) * real(signedWaveNumber(kz, f.nz)) / f.boxZ;
    const real qSquared = qx * qx + qy * qy + qz * qz;
    if (f.cutoff * std::sqrt(qSquared) > f.bandlimit)
    {
        return real(0);
    }
    const real volume = f.boxX * f.boxY * f.boxZ;
    return real(2.0 * M_PI) / (volume * f.bspX[kx] * f.bspY[ky] * f.bspZ[kz] * qSquared);
}

real triclinicAngularInfluenceReference(const SolveFixture& f, int kx, int ky, int kz)
{
    const real mx       = real(signedWaveNumber(kx, f.nx));
    const real my       = real(signedWaveNumber(ky, f.ny));
    const real mz       = real(signedWaveNumber(kz, f.nz));
    const real mhx      = mx * f.recipXX;
    const real mhy      = mx * f.recipYX + my * f.recipYY;
    const real mhz      = mx * f.recipZX + my * f.recipZY + mz * f.recipZZ;
    const real qSquared = real(4.0 * M_PI * M_PI) * (mhx * mhx + mhy * mhy + mhz * mhz);
    if (f.cutoff * std::sqrt(qSquared) > f.bandlimit)
    {
        return real(0);
    }
    return real(2.0 * M_PI) / (f.boxVolume * f.bspX[kx] * f.bspY[ky] * f.bspZ[kz] * qSquared);
}

TEST(EspSolve, InfluenceFunctionMatchesEq9)
{
    SolveFixture f;
    f.boxX = real(3.0);
    f.boxY = real(4.0);
    f.boxZ = real(5.0);
    f.run();

    const int kx = 2;
    const int ky = 1;
    const int kz = 0;
    EXPECT_NEAR(f.solver[f.index(kx, ky, kz)], angularInfluenceReference(f, kx, ky, kz), real(1e-6));
}

TEST(EspSolve, NormalizationCubicMatchesIntegerForm)
{
    SolveFixture f;
    f.boxX = f.boxY = f.boxZ = real(4.0);
    f.run();

    const int  kx          = 2;
    const int  ky          = 1;
    const int  kz          = 1;
    const real mSquared    = real(signedWaveNumber(kx, f.nx) * signedWaveNumber(kx, f.nx)
                               + signedWaveNumber(ky, f.ny) * signedWaveNumber(ky, f.ny)
                               + signedWaveNumber(kz, f.nz) * signedWaveNumber(kz, f.nz));
    const real integerForm = real(1.0) / (real(2.0 * M_PI) * f.boxX * mSquared);
    EXPECT_NEAR(f.solver[f.index(kx, ky, kz)], integerForm, real(1e-6));
}

TEST(EspSolve, KZeroIsExactlyZero)
{
    SolveFixture f;
    f.run();

    EXPECT_EQ(f.solver[f.index(0, 0, 0)], real(0));
}

TEST(EspSolve, KZeroNonIntegerSimdWidthTail)
{
    SolveFixture f;
    f.nx = 2 * GMX_SIMD_REAL_WIDTH + 3;
    f.ny = 3;
    f.nz = 2;
    f.resetStorage();
    f.run();

    EXPECT_EQ(f.solver[f.index(0, 0, 0)], real(0));
    EXPECT_NEAR(f.solver[f.index(f.nx - 1, 0, 0)], angularInfluenceReference(f, f.nx - 1, 0, 0), real(1e-6));
}

TEST(EspSolve, NegativeKxModesNotSkipped)
{
    SolveFixture f;
    f.run();

    const int kxPositive = 1;
    const int kxNegative = f.nx - 1;
    EXPECT_GT(f.solver[f.index(kxNegative, 0, 0)], real(0));
    EXPECT_NEAR(f.solver[f.index(kxNegative, 0, 0)], f.solver[f.index(kxPositive, 0, 0)], real(1e-6));
}

TEST(EspSolve, ChiHatZeroOutsideBandlimit)
{
    SolveFixture f;
    f.boxX = f.boxY = f.boxZ = real(1.0);
    f.bandlimit              = real(0.5);
    f.run();

    EXPECT_EQ(f.solver[f.index(1, 0, 0)], real(0));
}

TEST(EspSolve, OrthorhombicScalingMatchesAnalytical)
{
    SolveFixture f;
    f.boxX = real(2.5);
    f.boxY = real(4.5);
    f.boxZ = real(7.0);
    f.run();

    const int kx = 1;
    const int ky = 2;
    const int kz = 1;
    EXPECT_NEAR(f.solver[f.index(kx, ky, kz)], angularInfluenceReference(f, kx, ky, kz), real(1e-6));
}

TEST(EspSolve, TriclinicInfluenceUsesReciprocalBox)
{
    SolveFixture f;
    f.setTriclinicReciprocalBox(real(1.0) / real(3.0),
                                real(0.06),
                                real(1.0) / real(4.0),
                                real(-0.04),
                                real(0.03),
                                real(1.0) / real(5.0));
    f.run();

    const int kx = 2;
    const int ky = 1;
    const int kz = 1;
    EXPECT_NEAR(f.solver[f.index(kx, ky, kz)], triclinicAngularInfluenceReference(f, kx, ky, kz), real(1e-6));
}

TEST(EspSolve, StencilOrderAddsPmeGridScale)
{
    SolveFixture f;
    f.stencilOrder = 6;
    f.run();

    const int  kx        = 2;
    const int  ky        = 1;
    const int  kz        = 0;
    const real gridScale = real(0.5 * f.stencilOrder);
    const real expected  = angularInfluenceReference(f, kx, ky, kz) / (gridScale * gridScale);
    EXPECT_NEAR(f.solver[f.index(kx, ky, kz)], expected, real(1e-6));
}

} // namespace
} // namespace gmx::test
