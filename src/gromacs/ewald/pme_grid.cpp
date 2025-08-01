/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright 1991- The GROMACS Authors
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
/* TODO find out what this file should be called */
#include "gmxpre.h"

#include "pme_grid.h"

#include "config.h"

#include <cstdlib>

#include "gromacs/ewald/pme.h"
#include "gromacs/fft/parallel_3dfft.h"
#include "gromacs/math/vec.h"
#include "gromacs/timing/cyclecounter.h"
#include "gromacs/utility/fatalerror.h"

#include "pme_internal.h"

#ifdef DEBUG_PME
#    include "gromacs/fileio/pdbio.h"
#    include "gromacs/utility/cstringutil.h"
#    include "gromacs/utility/futil.h"
#endif

#include "pme_simd.h"

void gmx_sum_qgrid_dd(gmx_pme_t* pme, gmx::ArrayRef<real> grid, const int direction)
{
#if GMX_MPI
    pme_overlap_t* overlap;
    int            send_index0, send_nindex;
    int            recv_index0, recv_nindex;
    MPI_Status     stat;
    int            i, j, k, ix, iy, iz, icnt;
    int            send_id, recv_id, datasize;
    real*          p;
    real *         sendptr, *recvptr;

    /* Start with minor-rank communication. This is a bit of a pain since it is not contiguous */
    overlap = &pme->overlap[1];

    for (size_t ipulse = 0; ipulse < overlap->comm_data.size(); ipulse++)
    {
        /* Since we have already (un)wrapped the overlap in the z-dimension,
         * we only have to communicate 0 to nkz (not pmegrid_nz).
         */
        if (direction == GMX_SUM_GRID_FORWARD)
        {
            send_id     = overlap->comm_data[ipulse].send_id;
            recv_id     = overlap->comm_data[ipulse].recv_id;
            send_index0 = overlap->comm_data[ipulse].send_index0;
            send_nindex = overlap->comm_data[ipulse].send_nindex;
            recv_index0 = overlap->comm_data[ipulse].recv_index0;
            recv_nindex = overlap->comm_data[ipulse].recv_nindex;
        }
        else
        {
            send_id     = overlap->comm_data[ipulse].recv_id;
            recv_id     = overlap->comm_data[ipulse].send_id;
            send_index0 = overlap->comm_data[ipulse].recv_index0;
            send_nindex = overlap->comm_data[ipulse].recv_nindex;
            recv_index0 = overlap->comm_data[ipulse].send_index0;
            recv_nindex = overlap->comm_data[ipulse].send_nindex;
        }

        /* Copy data to contiguous send buffer */
        if (debug)
        {
            fprintf(debug,
                    "PME send rank %d %d -> %d grid start %d Communicating %d to %d\n",
                    pme->nodeid,
                    overlap->nodeid,
                    send_id,
                    pme->pmegrid_start_iy,
                    send_index0 - pme->pmegrid_start_iy,
                    send_index0 - pme->pmegrid_start_iy + send_nindex);
        }
        icnt = 0;
        for (i = 0; i < pme->pmegrid_nx; i++)
        {
            ix = i;
            for (j = 0; j < send_nindex; j++)
            {
                iy = j + send_index0 - pme->pmegrid_start_iy;
                for (k = 0; k < pme->nkz; k++)
                {
                    iz = k;
                    overlap->sendbuf[icnt++] =
                            grid[ix * (pme->pmegrid_ny * pme->pmegrid_nz) + iy * (pme->pmegrid_nz) + iz];
                }
            }
        }

        datasize = pme->pmegrid_nx * pme->nkz;

        MPI_Sendrecv(overlap->sendbuf.data(),
                     send_nindex * datasize,
                     GMX_MPI_REAL,
                     send_id,
                     ipulse,
                     overlap->recvbuf.data(),
                     recv_nindex * datasize,
                     GMX_MPI_REAL,
                     recv_id,
                     ipulse,
                     overlap->mpi_comm,
                     &stat);

        /* Get data from contiguous recv buffer */
        if (debug)
        {
            fprintf(debug,
                    "PME recv rank %d %d <- %d grid start %d Communicating %d to %d\n",
                    pme->nodeid,
                    overlap->nodeid,
                    recv_id,
                    pme->pmegrid_start_iy,
                    recv_index0 - pme->pmegrid_start_iy,
                    recv_index0 - pme->pmegrid_start_iy + recv_nindex);
        }
        icnt = 0;
        for (i = 0; i < pme->pmegrid_nx; i++)
        {
            ix = i;
            for (j = 0; j < recv_nindex; j++)
            {
                iy = j + recv_index0 - pme->pmegrid_start_iy;
                for (k = 0; k < pme->nkz; k++)
                {
                    iz = k;
                    if (direction == GMX_SUM_GRID_FORWARD)
                    {
                        grid[ix * (pme->pmegrid_ny * pme->pmegrid_nz) + iy * (pme->pmegrid_nz) + iz] +=
                                overlap->recvbuf[icnt++];
                    }
                    else
                    {
                        grid[ix * (pme->pmegrid_ny * pme->pmegrid_nz) + iy * (pme->pmegrid_nz) + iz] =
                                overlap->recvbuf[icnt++];
                    }
                }
            }
        }
    }

    /* Major dimension is easier, no copying required,
     * but we might have to sum to separate array.
     * Since we don't copy, we have to communicate up to pmegrid_nz,
     * not nkz as for the minor direction.
     */
    overlap = &pme->overlap[0];

    for (size_t ipulse = 0; ipulse < overlap->comm_data.size(); ipulse++)
    {
        if (direction == GMX_SUM_GRID_FORWARD)
        {
            send_id     = overlap->comm_data[ipulse].send_id;
            recv_id     = overlap->comm_data[ipulse].recv_id;
            send_index0 = overlap->comm_data[ipulse].send_index0;
            send_nindex = overlap->comm_data[ipulse].send_nindex;
            recv_index0 = overlap->comm_data[ipulse].recv_index0;
            recv_nindex = overlap->comm_data[ipulse].recv_nindex;
            recvptr     = overlap->recvbuf.data();
        }
        else
        {
            send_id     = overlap->comm_data[ipulse].recv_id;
            recv_id     = overlap->comm_data[ipulse].send_id;
            send_index0 = overlap->comm_data[ipulse].recv_index0;
            send_nindex = overlap->comm_data[ipulse].recv_nindex;
            recv_index0 = overlap->comm_data[ipulse].send_index0;
            recv_nindex = overlap->comm_data[ipulse].send_nindex;
            recvptr     = grid.data()
                      + (recv_index0 - pme->pmegrid_start_ix) * (pme->pmegrid_ny * pme->pmegrid_nz);
        }

        sendptr = grid.data() + (send_index0 - pme->pmegrid_start_ix) * (pme->pmegrid_ny * pme->pmegrid_nz);
        datasize = pme->pmegrid_ny * pme->pmegrid_nz;

        if (debug)
        {
            fprintf(debug,
                    "PME send rank %d %d -> %d grid start %d Communicating %d to %d\n",
                    pme->nodeid,
                    overlap->nodeid,
                    send_id,
                    pme->pmegrid_start_ix,
                    send_index0 - pme->pmegrid_start_ix,
                    send_index0 - pme->pmegrid_start_ix + send_nindex);
            fprintf(debug,
                    "PME recv rank %d %d <- %d grid start %d Communicating %d to %d\n",
                    pme->nodeid,
                    overlap->nodeid,
                    recv_id,
                    pme->pmegrid_start_ix,
                    recv_index0 - pme->pmegrid_start_ix,
                    recv_index0 - pme->pmegrid_start_ix + recv_nindex);
        }

        MPI_Sendrecv(sendptr,
                     send_nindex * datasize,
                     GMX_MPI_REAL,
                     send_id,
                     ipulse,
                     recvptr,
                     recv_nindex * datasize,
                     GMX_MPI_REAL,
                     recv_id,
                     ipulse,
                     overlap->mpi_comm,
                     &stat);

        /* ADD data from contiguous recv buffer */
        if (direction == GMX_SUM_GRID_FORWARD)
        {
            p = grid.data() + (recv_index0 - pme->pmegrid_start_ix) * (pme->pmegrid_ny * pme->pmegrid_nz);
            for (i = 0; i < recv_nindex * datasize; i++)
            {
                p[i] += overlap->recvbuf[i];
            }
        }
    }
#else  // GMX_MPI
    GMX_UNUSED_VALUE(pme);
    GMX_UNUSED_VALUE(grid);
    GMX_UNUSED_VALUE(direction);

    GMX_RELEASE_ASSERT(false, "gmx_sum_qgrid_dd() should not be called without MPI");
#endif // GMX_MPI
}


int copy_pmegrid_to_fftgrid(const gmx_pme_t* pme, PmeAndFftGrids* grids)
{
    const real* gmx_restrict pmegrid = grids->pmeGrids.grid.grid.data();
    real* gmx_restrict       fftgrid = grids->fftgrid;

    ivec local_fft_ndata, local_fft_offset, local_fft_size;
    ivec local_pme_size;
    int  ix, iy, iz;
    int  pmeidx, fftidx;

    /* Dimensions should be identical for A/B grid, so we just use A here */
    gmx_parallel_3dfft_real_limits(
            grids->pfft_setup.get(), local_fft_ndata, local_fft_offset, local_fft_size);

    local_pme_size[0] = pme->pmegrid_nx;
    local_pme_size[1] = pme->pmegrid_ny;
    local_pme_size[2] = pme->pmegrid_nz;

    /* The fftgrid is always 'justified' to the lower-left corner of the PME grid,
       the offset is identical, and the PME grid always has more data (due to overlap)
     */
    {
#ifdef DEBUG_PME
        FILE *fp, *fp2;
        char  fn[STRLEN];
        real  val;
        sprintf(fn, "pmegrid%d.pdb", pme->nodeid);
        fp = gmx_ffopen(fn, "w");
        sprintf(fn, "pmegrid%d.txt", pme->nodeid);
        fp2 = gmx_ffopen(fn, "w");
#endif

        for (ix = 0; ix < local_fft_ndata[XX]; ix++)
        {
            for (iy = 0; iy < local_fft_ndata[YY]; iy++)
            {
                for (iz = 0; iz < local_fft_ndata[ZZ]; iz++)
                {
                    pmeidx = ix * (local_pme_size[YY] * local_pme_size[ZZ])
                             + iy * (local_pme_size[ZZ]) + iz;
                    fftidx = ix * (local_fft_size[YY] * local_fft_size[ZZ])
                             + iy * (local_fft_size[ZZ]) + iz;
                    fftgrid[fftidx] = pmegrid[pmeidx];
#ifdef DEBUG_PME
                    val = 100 * pmegrid[pmeidx];
                    if (pmegrid[pmeidx] != 0)
                    {
                        gmx_fprintf_pdb_atomline(fp,
                                                 epdbATOM,
                                                 pmeidx,
                                                 "CA",
                                                 ' ',
                                                 "GLY",
                                                 ' ',
                                                 pmeidx,
                                                 ' ',
                                                 5.0 * ix,
                                                 5.0 * iy,
                                                 5.0 * iz,
                                                 1.0,
                                                 val,
                                                 "");
                    }
                    if (pmegrid[pmeidx] != 0)
                    {
                        fprintf(fp2,
                                "%-12s  %5d  %5d  %5d  %12.5e\n",
                                "qgrid",
                                pme->pmegrid_start_ix + ix,
                                pme->pmegrid_start_iy + iy,
                                pme->pmegrid_start_iz + iz,
                                pmegrid[pmeidx]);
                    }
#endif
                }
            }
        }
#ifdef DEBUG_PME
        gmx_ffclose(fp);
        gmx_ffclose(fp2);
#endif
    }
    return 0;
}


#ifdef PME_TIME_THREADS
static gmx_cycles_t omp_cyc_start()
{
    return gmx_cycles_read();
}

static gmx_cycles_t omp_cyc_end(gmx_cycles_t c)
{
    return gmx_cycles_read() - c;
}
#endif


int copy_fftgrid_to_pmegrid(const gmx_pme_t* pme, PmeAndFftGrids* grids, int nthread, int thread)
{
    const real* gmx_restrict fftgrid = grids->fftgrid;
    real* gmx_restrict       pmegrid = grids->pmeGrids.grid.grid.data();

    ivec local_fft_ndata, local_fft_offset, local_fft_size;
    ivec local_pme_size;
    int  ixy0, ixy1, ixy, ix, iy, iz;
    int  pmeidx, fftidx;
#ifdef PME_TIME_THREADS
    gmx_cycles_t  c1;
    static double cs1 = 0;
    static int    cnt = 0;
#endif

#ifdef PME_TIME_THREADS
    c1 = omp_cyc_start();
#endif
    /* Dimensions should be identical for A/B grid, so we just use A here */
    gmx_parallel_3dfft_real_limits(
            grids->pfft_setup.get(), local_fft_ndata, local_fft_offset, local_fft_size);

    local_pme_size[0] = pme->pmegrid_nx;
    local_pme_size[1] = pme->pmegrid_ny;
    local_pme_size[2] = pme->pmegrid_nz;

    /* The fftgrid is always 'justified' to the lower-left corner of the PME grid,
       the offset is identical, and the PME grid always has more data (due to overlap)
     */
    ixy0 = ((thread)*local_fft_ndata[XX] * local_fft_ndata[YY]) / nthread;
    ixy1 = ((thread + 1) * local_fft_ndata[XX] * local_fft_ndata[YY]) / nthread;

    for (ixy = ixy0; ixy < ixy1; ixy++)
    {
        ix = ixy / local_fft_ndata[YY];
        iy = ixy - ix * local_fft_ndata[YY];

        pmeidx = (ix * local_pme_size[YY] + iy) * local_pme_size[ZZ];
        fftidx = (ix * local_fft_size[YY] + iy) * local_fft_size[ZZ];
        for (iz = 0; iz < local_fft_ndata[ZZ]; iz++)
        {
            pmegrid[pmeidx + iz] = fftgrid[fftidx + iz];
        }
    }

#ifdef PME_TIME_THREADS
    c1 = omp_cyc_end(c1);
    cs1 += (double)c1;
    cnt++;
    if (cnt % 20 == 0)
    {
        printf("copy %.2f\n", cs1 * 1e-9);
    }
#endif

    return 0;
}


void wrap_periodic_pmegrid(const gmx_pme_t* pme, gmx::ArrayRef<real> pmegrid)
{
    int nx, ny, nz, pny, pnz, ny_x, overlap, ix, iy, iz;

    nx = pme->nkx;
    ny = pme->nky;
    nz = pme->nkz;

    pny = pme->pmegrid_ny;
    pnz = pme->pmegrid_nz;

    overlap = pme->pme_order - 1;

    /* Add periodic overlap in z */
    for (ix = 0; ix < pme->pmegrid_nx; ix++)
    {
        for (iy = 0; iy < pme->pmegrid_ny; iy++)
        {
            for (iz = 0; iz < overlap; iz++)
            {
                pmegrid[(ix * pny + iy) * pnz + iz] += pmegrid[(ix * pny + iy) * pnz + nz + iz];
            }
        }
    }

    if (pme->nnodes_minor == 1)
    {
        for (ix = 0; ix < pme->pmegrid_nx; ix++)
        {
            for (iy = 0; iy < overlap; iy++)
            {
                for (iz = 0; iz < nz; iz++)
                {
                    pmegrid[(ix * pny + iy) * pnz + iz] += pmegrid[(ix * pny + ny + iy) * pnz + iz];
                }
            }
        }
    }

    if (pme->nnodes_major == 1)
    {
        ny_x = (pme->nnodes_minor == 1 ? ny : pme->pmegrid_ny);

        for (ix = 0; ix < overlap; ix++)
        {
            for (iy = 0; iy < ny_x; iy++)
            {
                for (iz = 0; iz < nz; iz++)
                {
                    pmegrid[(ix * pny + iy) * pnz + iz] += pmegrid[((nx + ix) * pny + iy) * pnz + iz];
                }
            }
        }
    }
}


void unwrap_periodic_pmegrid(gmx_pme_t* pme, gmx::ArrayRef<real> pmegrid)
{
    int nx, ny, nz, pny, pnz, ny_x, overlap, ix;

    nx = pme->nkx;
    ny = pme->nky;
    nz = pme->nkz;

    pny = pme->pmegrid_ny;
    pnz = pme->pmegrid_nz;

    overlap = pme->pme_order - 1;

    if (pme->nnodes_major == 1)
    {
        ny_x = (pme->nnodes_minor == 1 ? ny : pme->pmegrid_ny);

        for (ix = 0; ix < overlap; ix++)
        {
            int iy, iz;

            for (iy = 0; iy < ny_x; iy++)
            {
                for (iz = 0; iz < nz; iz++)
                {
                    pmegrid[((nx + ix) * pny + iy) * pnz + iz] = pmegrid[(ix * pny + iy) * pnz + iz];
                }
            }
        }
    }

    if (pme->nnodes_minor == 1)
    {
#pragma omp parallel for num_threads(pme->nthread) schedule(static)
        for (ix = 0; ix < pme->pmegrid_nx; ix++)
        {
            // Trivial OpenMP region that does not throw, no need for try/catch
            int iy, iz;

            for (iy = 0; iy < overlap; iy++)
            {
                for (iz = 0; iz < nz; iz++)
                {
                    pmegrid[(ix * pny + ny + iy) * pnz + iz] = pmegrid[(ix * pny + iy) * pnz + iz];
                }
            }
        }
    }

    /* Copy periodic overlap in z */
#pragma omp parallel for num_threads(pme->nthread) schedule(static)
    for (ix = 0; ix < pme->pmegrid_nx; ix++)
    {
        // Trivial OpenMP region that does not throw, no need for try/catch
        int iy, iz;

        for (iy = 0; iy < pme->pmegrid_ny; iy++)
        {
            for (iz = 0; iz < overlap; iz++)
            {
                pmegrid[(ix * pny + iy) * pnz + nz + iz] = pmegrid[(ix * pny + iy) * pnz + iz];
            }
        }
    }
}

void set_grid_alignment(int gmx_unused* pmegrid_nz, int gmx_unused pme_order)
{
#ifdef PME_SIMD4_SPREAD_GATHER
    if (pme_order == 5
#    if !PME_4NSIMD_GATHER
        || pme_order == 4
#    endif
    )
    {
        /* Round nz up to a multiple of 4 to ensure alignment */
        *pmegrid_nz = ((*pmegrid_nz + 3) & ~3);
    }
#endif
}

static void pmegrid_init(pmegrid_t*           grid,
                         int                  cx,
                         int                  cy,
                         int                  cz,
                         int                  x0,
                         int                  y0,
                         int                  z0,
                         int                  x1,
                         int                  y1,
                         int                  z1,
                         gmx_bool             set_alignment,
                         int                  pme_order,
                         AlignedVector<real>* gridStorage)
{
    GMX_RELEASE_ASSERT(gridStorage != nullptr, "We need storage");

    int nz, gridsize;

    grid->ci[XX]     = cx;
    grid->ci[YY]     = cy;
    grid->ci[ZZ]     = cz;
    grid->offset[XX] = x0;
    grid->offset[YY] = y0;
    grid->offset[ZZ] = z0;
    grid->n[XX]      = x1 - x0 + pme_order - 1;
    grid->n[YY]      = y1 - y0 + pme_order - 1;
    grid->n[ZZ]      = z1 - z0 + pme_order - 1;
    copy_ivec(grid->n, grid->s);

    nz = grid->s[ZZ];
    set_grid_alignment(&nz, pme_order);
    if (set_alignment)
    {
        grid->s[ZZ] = nz;
    }
    else if (nz != grid->s[ZZ])
    {
        gmx_incons("pmegrid_init call with an unaligned z size");
    }

    grid->order = pme_order;
    gridsize    = grid->s[XX] * grid->s[YY] * grid->s[ZZ];

    GMX_RELEASE_ASSERT(gridStorage != nullptr, "Need storage");

    if (gridStorage->empty())
    {
        // Allocate new padded memory and point grid to that
        gridStorage->resize(gridsize);

        grid->grid = *gridStorage;
    }
    else
    {
        // Should we increase the storage?
        if (gridStorage->size() < static_cast<size_t>(gridsize))
        {
            // Resize the storage to fit the grid size
            gridStorage->resize(gridsize);
        }

        // Use already allocated memory
        gmx::ArrayRef<real> memoryView = *gridStorage;

        GMX_RELEASE_ASSERT(memoryView.ssize() >= gridsize,
                           "memoryView should be sufficiently large");

        // When using aligned SIMD4 operations, check the alignment of the memory
#ifdef PME_SIMD4_SPREAD_GATHER
        if (pme_order == 5
#    if !PME_4NSIMD_GATHER
            || pme_order == 4
#    endif
        )
        {
            GMX_RELEASE_ASSERT(
                    reinterpret_cast<std::uintptr_t>(memoryView.data()) % (4 * sizeof(real)) == 0,
                    "Start of memoryView should be SIMD4 aligned");
        }
#endif

        // Set the memory view
        grid->grid = memoryView.subArray(0, gridsize);
    }
}

static void make_subgrid_division(const ivec n, int ovl, int nthread, ivec nsub)
{
    int   gsize_opt, gsize;
    int   nsx, nsy, nsz;
    char* env;

    gsize_opt = -1;
    for (nsx = 1; nsx <= nthread; nsx++)
    {
        if (nthread % nsx == 0)
        {
            for (nsy = 1; nsy <= nthread; nsy++)
            {
                if (nsx * nsy <= nthread && nthread % (nsx * nsy) == 0)
                {
                    nsz = nthread / (nsx * nsy);

                    /* Determine the number of grid points per thread */
                    gsize = (gmx::divideRoundUp(n[XX], nsx) + ovl) * (gmx::divideRoundUp(n[YY], nsy) + ovl)
                            * (gmx::divideRoundUp(n[ZZ], nsz) + ovl);

                    /* Minimize the number of grids points per thread
                     * and, secondarily, the number of cuts in minor dimensions.
                     */
                    if (gsize_opt == -1 || gsize < gsize_opt
                        || (gsize == gsize_opt && (nsz < nsub[ZZ] || (nsz == nsub[ZZ] && nsy < nsub[YY]))))
                    {
                        nsub[XX]  = nsx;
                        nsub[YY]  = nsy;
                        nsub[ZZ]  = nsz;
                        gsize_opt = gsize;
                    }
                }
            }
        }
    }

    env = std::getenv("GMX_PME_THREAD_DIVISION");
    if (env != nullptr)
    {
        sscanf(env, "%20d %20d %20d", &nsub[XX], &nsub[YY], &nsub[ZZ]);
    }

    if (nsub[XX] * nsub[YY] * nsub[ZZ] != nthread)
    {
        gmx_fatal(FARGS,
                  "PME grid thread division (%d x %d x %d) does not match the total number of "
                  "threads (%d)",
                  nsub[XX],
                  nsub[YY],
                  nsub[ZZ],
                  nthread);
    }
}

void pmegrids_init(pmegrids_t*                        grids,
                   int                                nx,
                   int                                ny,
                   int                                nz,
                   int                                nz_base,
                   int                                pme_order,
                   gmx_bool                           bUseThreads,
                   int                                nthread,
                   int                                overlap_x,
                   int                                overlap_y,
                   gmx::ArrayRef<AlignedVector<real>> gridsStorage)
{
    GMX_RELEASE_ASSERT(!gridsStorage.empty(), "Need storage");

    ivec n, n_base;
    int  i, tfac;
    int  max_comm_lines = -1;

    n[XX] = nx - (pme_order - 1);
    n[YY] = ny - (pme_order - 1);
    n[ZZ] = nz - (pme_order - 1);

    copy_ivec(n, n_base);
    n_base[ZZ] = nz_base;

    pmegrid_init(&grids->grid, 0, 0, 0, 0, 0, 0, n[XX], n[YY], n[ZZ], FALSE, pme_order, &gridsStorage[0]);

    grids->nthread = nthread;

    make_subgrid_division(n_base, pme_order - 1, grids->nthread, grids->nc);

    if (bUseThreads)
    {
        GMX_RELEASE_ASSERT(gridsStorage.ssize() == 1 + nthread,
                           "Expect 1 + #thread grids in the storage");

        ivec nst;

        for (int d = 0; d < DIM; d++)
        {
            nst[d] = gmx::divideRoundUp(n[d], grids->nc[d]) + pme_order - 1;
        }
        set_grid_alignment(&nst[ZZ], pme_order);

        if (debug)
        {
            fprintf(debug,
                    "pmegrid thread local division: %d x %d x %d\n",
                    grids->nc[XX],
                    grids->nc[YY],
                    grids->nc[ZZ]);
            fprintf(debug, "pmegrid %d %d %d max thread pmegrid %d %d %d\n", nx, ny, nz, nst[XX], nst[YY], nst[ZZ]);
        }

        grids->grid_th.resize(grids->nthread);

        const int threadGridSize = nst[XX] * nst[YY] * nst[ZZ];

#pragma omp parallel for num_threads(nthread) schedule(static)
        for (int thread = 0; thread < nthread; thread++)
        {
            const int x = thread / (grids->nc[YY] * grids->nc[ZZ]);
            const int y = (thread / grids->nc[ZZ]) % grids->nc[YY];
            const int z = thread % grids->nc[ZZ];

            AlignedVector<real>& gridStorage = gridsStorage[1 + thread];
            if (gridStorage.empty())
            {
                gridStorage.resize(threadGridSize);
            }
            else
            {
                GMX_RELEASE_ASSERT(gmx::ssize(gridStorage) >= threadGridSize,
                                   "Passed storage should be sufficiently large");
            }

            pmegrid_init(&grids->grid_th[thread],
                         x,
                         y,
                         z,
                         (n[XX] * (x)) / grids->nc[XX],
                         (n[YY] * (y)) / grids->nc[YY],
                         (n[ZZ] * (z)) / grids->nc[ZZ],
                         (n[XX] * (x + 1)) / grids->nc[XX],
                         (n[YY] * (y + 1)) / grids->nc[YY],
                         (n[ZZ] * (z + 1)) / grids->nc[ZZ],
                         TRUE,
                         pme_order,
                         &gridStorage);
        }
    }

    tfac = 1;
    for (int d = DIM - 1; d >= 0; d--)
    {
        grids->g2t[d].resize(n[d]);
        int t = 0;
        for (i = 0; i < n[d]; i++)
        {
            /* The second check should match the parameters
             * of the pmegrid_init call above.
             */
            while (t + 1 < grids->nc[d] && i >= (n[d] * (t + 1)) / grids->nc[d])
            {
                t++;
            }
            grids->g2t[d][i] = t * tfac;
        }

        tfac *= grids->nc[d];

        switch (d)
        {
            case XX: max_comm_lines = overlap_x; break;
            case YY: max_comm_lines = overlap_y; break;
            case ZZ: max_comm_lines = pme_order - 1; break;
        }
        grids->nthread_comm[d] = 0;
        while ((n[d] * grids->nthread_comm[d]) / grids->nc[d] < max_comm_lines
               && grids->nthread_comm[d] < grids->nc[d])
        {
            grids->nthread_comm[d]++;
        }
        if (debug != nullptr)
        {
            fprintf(debug,
                    "pmegrid thread grid communication range in %c: %d\n",
                    'x' + d,
                    grids->nthread_comm[d]);
        }
        /* It should be possible to make grids->nthread_comm[d]==grids->nc[d]
         * work, but this is not a problematic restriction.
         */
        if (grids->nc[d] > 1 && grids->nthread_comm[d] > grids->nc[d])
        {
            gmx_fatal(FARGS,
                      "Too many threads for PME (%d) compared to the number of grid lines, reduce "
                      "the number of threads doing PME",
                      grids->nthread);
        }
    }
}

std::tuple<std::vector<int>, std::vector<real>>
make_gridindex_to_localindex(int n, int local_start, int local_range, bool checkRoundingAtBoundary)
{
    /* Here we construct array for looking up the grid line index and
     * fraction for particles. This is done because it is slighlty
     * faster than the modulo operation and to because we need to take
     * care of rounding issues, see below.
     * We use an array size of c_pmeNeighborUnitcellCount times the grid size
     * to allow for particles to be out of the triclinic unit-cell.
     */
    const int         arraySize = c_pmeNeighborUnitcellCount * n;
    std::vector<int>  gtl(arraySize);
    std::vector<real> fsh(arraySize);

    for (int i = 0; i < arraySize; i++)
    {
        /* Transform global grid index to the local grid index.
         * Our local grid always runs from 0 to local_range-1.
         */
        gtl[i] = (i - local_start + n) % n;
        /* For coordinates that fall within the local grid the fraction
         * is correct, we don't need to shift it.
         */
        fsh[i] = 0;
        /* Check if we are using domain decomposition for PME */
        if (local_range < n && checkRoundingAtBoundary)
        {
            /* Due to rounding issues i could be 1 beyond the lower or
             * upper boundary of the local grid. Correct the index for this.
             * If we shift the index, we need to shift the fraction by
             * the same amount in the other direction to not affect
             * the weights.
             * Note that due to this shifting the weights at the end of
             * the spline might change, but that will only involve values
             * between zero and values close to the precision of a real,
             * which is anyhow the accuracy of the whole mesh calculation.
             */
            if (gtl[i] == n - 1)
            {
                /* When this i is used, we should round the local index up */
                gtl[i] = 0;
                fsh[i] = -1;
            }
            else if (gtl[i] == local_range && local_range > 0)
            {
                /* When this i is used, we should round the local index down */
                gtl[i] = local_range - 1;
                fsh[i] = 1;
            }
        }
    }

    return { std::move(gtl), std::move(fsh) };
}
