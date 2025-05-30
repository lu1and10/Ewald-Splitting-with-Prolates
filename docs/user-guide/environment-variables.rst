.. NOTE: Below is a useful bash one-liner to verify whether there are variables in this file
..        no longer present in the code.
.. ( export INPUT_FILE='docs/user-guide/environment-variables.rst' GIT_PAGER="cat "; for s in $(grep '^`'  $INPUT_FILE | sed 's/`//g' | sed 's/,/ /g'); do count=$(git grep $s | grep -v $INPUT_FILE | wc -l); [ $count -eq 0 ] && printf "%-30s%s\n" $s $count; done ; )
.. Another useful one-liner to find undocumented variables:
..  ( export INPUT_FILE=docs/user-guide/environment-variables.rst; GIT_PAGER="cat ";   for ss in `for s in $(git grep getenv |  sed 's/.*getenv("\(.*\)".*/\1/' | sort -u  | grep '^[A-Z]'); do [ $(grep $s $INPUT_FILE -c) -eq 0 ] && echo $s; done `; do git grep $ss ; done )

Environment Variables
=====================

|Gromacs| programs may be influenced by the use of
environment variables.  First of all, the variables set in
the ``GMXRC`` file are essential for running and
compiling |Gromacs|. Some other useful environment variables are
listed in the following sections. Most environment variables function
by being set in your shell to any non-NULL value. Specific
requirements are described below if other values need to be set. You
should consult the documentation for your shell for instructions on
how to set environment variables in the current shell, or in configuration
files for future shells. Note that requirements for exporting
environment variables to jobs run under batch control systems vary and
you should consult your local documentation for details.

Output Control
--------------

..
   Please keep these in alphabetical order!

``GMX_COMPELDUMP``
        Applies for computational electrophysiology setups
        only (see reference manual). The initial structure gets dumped to
        :ref:`pdb` file, which allows to check whether multimeric channels have
        the correct PBC representation.

``GMX_DISABLE_GPU_TIMING``
        Disables GPU timings in the log file for OpenCL.

``GMX_ENABLE_GPU_TIMING``
        Enables GPU timings in the log file for CUDA and SYCL. Note that CUDA
        timings are incorrect with multiple streams, as happens with domain
        decomposition or with both non-bondeds and PME on the GPU (this is
        also the main reason why they are not turned on by default).

``GMX_LOG_BUFFER``
        the size of the buffer for file I/O. When set
        to 0, all file I/O will be unbuffered and therefore very slow.
        This can be handy for debugging purposes, because it ensures
        that all files are always totally up-to-date.

``GMX_MAXBACKUP``
        |Gromacs| automatically backs up old
        copies of files when trying to write a new file of the same
        name, and this variable controls the maximum number of
        backups that will be made, default 99. If set to 0 it fails to
        run if any output file already exists. And if set to -1 it
        overwrites any output file without making a backup.

``GMX_NO_QUOTES``
        if this is explicitly set, no cool quotes
        will be printed at the end of a program.

``GMX_PRINT_LONGFORMAT``
        use long float format when printing
        decimal values.

``GMX_SUPPRESS_DUMP``
        prevent dumping of step files during
        (for example) blowing up during failure of constraint
        algorithms.

``GMX_TPI_DUMP``
        dump all configurations to a :ref:`pdb`
        file that have an interaction energy less than the value set
        in this environment variable.

``GMX_TRAJECTORY_IO_VERBOSITY``
        Defaults to 1, which prints frame count e.g. when reading trajectory
        files. Set to 0 for quiet operation.

``GMX_VIEW_XVG``
        ``GMX_VIEW_EPS`` and ``GMX_VIEW_PDB``, commands used to
        automatically view :ref:`xvg`, :ref:`eps`
        and :ref:`pdb` file types, respectively; they default to ``xmgrace``,
        ``ghostview`` and ``rasmol``. Set to empty to disable
        automatic viewing of a particular file type. The command will
        be forked off and run in the background at the same priority
        as the |Gromacs| tool (which might not be what you want).
        Be careful not to use a command which blocks the terminal
        (e.g. ``vi``), since multiple instances might be run.

Debugging
---------

..
   Please keep these in alphabetical order!

``GMX_DD_NPULSE``
        over-ride the number of DD pulses used
        (default 0, meaning no over-ride). Normally 1 or 2.

``GMX_DD_DEBUG``
        general debugging trigger for every domain
        decomposition (default 0, meaning off). Currently only checks
        global-local atom index mapping for consistency.

``GMX_DD_NST_DUMP``
        number of steps that elapse between dumping
        the current DD to a PDB file (default 0). This only takes effect
        during domain decomposition, so it should typically be
        0 (never), 1 (every DD phase) or a multiple of :mdp:`nstlist`.

``GMX_DD_NST_DUMP_GRID``
        number of steps that elapse between dumping
        the current DD grid to a PDB file (default 0). This only takes effect
        during domain decomposition, so it should typically be
        0 (never), 1 (every DD phase) or a multiple of :mdp:`nstlist`.

``GMX_DISABLE_ALTERNATING_GPU_WAIT``
        disables the specialized polling wait path used to wait for the PME and nonbonded
        GPU tasks completion to overlap to do the reduction of the resulting forces that
        arrive first. Setting this variable switches to the generic path with fixed waiting
        order.

``GMX_TEST_REQUIRED_NUMBER_OF_DEVICES``
        sets the number of GPUs required by the test suite. By default, the test suite would
        fall-back to using CPU if GPUs could not be detected. Set it to a positive integer value
        to ensure that at least this at least this number of usable GPUs are detected. Default:
        0 (not testing GPU availability).

There are a number of extra environment variables like these
that are used in debugging - check the code!

Performance and Run Control
---------------------------

..
   Please keep these in alphabetical order!

``GMX_AWH_NO_POINT_LIMIT``
        Removes the upper limit on the number of points in an AWH bias grid.
        By default, an error is raised if the grid is unreasonably large and
        can cause sampling problems. Setting this variable will only remove
        this safety check. It is recommended instead to reduce the grid size,
        e.g., by using lower force constants.

``GMX_BONDED_NTHREAD_UNIFORM``
        Value of the number of threads per rank from which to switch from uniform
        to localized bonded interaction distribution; optimal value dependent on
        system and hardware, default value is 4.

``GMX_CUDA_GRAPH``
        Use CUDA Graphs to schedule a graph on each step rather than multiple
        activities scheduled to multiple CUDA streams, if the run conditions allow. Experimental.

``GMX_CYCLE_ALL``
        times all code during runs.  Incompatible with threads.

``GMX_CYCLE_BARRIER``
        calls MPI_Barrier before each cycle start/stop call.

``GMX_DD_ORDER_ZYX``
        build domain decomposition cells in the order
        (z, y, x) rather than the default (x, y, z).

``GMX_DD_RECORD_LOAD``
        record DD load statistics for reporting at end of the run (default 1, meaning on)

``GMX_DD_SINGLE_RANK``
        Controls the use of the domain decomposition machinery when using a single MPI rank.
        Value 0 turns DD off, 1 turns DD on. Default is automated choice based on heuristics.

``GMX_DD_USE_SENDRECV2``
        during constraint and vsite communication, use a pair
        of ``MPI_Sendrecv`` calls instead of two simultaneous non-blocking calls
        (default 0, meaning off). Might be faster on some MPI implementations.

``GMX_DETAILED_PERF_STATS``
        when set, print slightly more detailed performance information
        to the :ref:`log` file. The resulting output is the way performance summary is reported in versions
        4.5.x and thus may be useful for anyone using scripts to parse :ref:`log` files or standard output.

``GMX_DISABLE_CUDA_TIMING``
        Deprecated. Use ``GMX_DISABLE_GPU_TIMING`` instead.

``GMX_DISABLE_DYNAMICPRUNING``
        disables dynamic pair-list pruning. Note that :ref:`gmx mdrun` will
        still tune nstlist to the optimal value picked assuming dynamic pruning. Thus
        for good performance the -nstlist option should be used.

``GMX_DISABLE_GPU_DETECTION``
        when set, disables GPU detection even if :ref:`gmx mdrun` was compiled
        with GPU support.

``GMX_DISABLE_GPU_TIMING``
        timing of asynchronously executed GPU operations can have a
        non-negligible overhead with short step times. Disabling timing can improve performance in these cases.
        Timings are disabled by default with CUDA and SYCL.

``GMX_DISABLE_SIMD_KERNELS``
        disables architecture-specific SIMD-optimized (SSE2, SSE4.1, AVX, etc.)
        non-bonded kernels thus forcing the use of plain C kernels.

``GMX_DISABLE_STAGED_GPU_TO_CPU_PMEPP_COMM``
        Use direct rather than staged GPU communications for PME force
        transfers from the PME GPU to the CPU memory of a PP
        rank. This may have advantages in PCIe-only servers, or for
        runs with low atom counts (which are more sensitive to latency
        than bandwidth).

``GMX_DISRE_ENSEMBLE_SIZE``
        the number of systems for distance restraint ensemble
        averaging. Takes an integer value.

``GMX_DLB_BASED_ON_FLOPS``
        do domain-decomposition dynamic load balancing based on flop count rather than
        measured time elapsed (default 0, meaning off).
        This makes the load balancing reproducible, which can be useful for debugging purposes.
        A value of 1 uses the flops; a value > 1 adds (value - 1)*5% of noise to the flops to increase the imbalance and the scaling.

``GMX_DLB_MAX_BOX_SCALING``
        maximum percentage box scaling permitted per domain-decomposition
        load-balancing step (default 10)

``GMX_EMULATE_GPU``
        emulate GPU runs by using algorithmically equivalent CPU reference code instead of
        GPU-accelerated functions. As the CPU code is slow, it is intended to be used only for debugging purposes.

``GMX_DISABLE_DIRECT_GPU_COMM``
        Disables the direct GPU communication in multi-rank parallel runs
        even when build and simulation setup support it.

``GMX_ENABLE_NVSHMEM``
        Enables GPU kernel-initiated communication using NVSHMEM in multi-rank parallel runs
        when build and simulation setup support it.

``GMX_ENABLE_STAGED_GPU_TO_CPU_PMEPP_COMM``
        Use a staged implementation of GPU communications for PME force
        transfers from the PME GPU to the CPU memory of a PP rank for
        thread-MPI. The staging is done via a GPU buffer on the PP
        GPU. This is expected to be beneficial for servers with direct
        communication links between GPUs.

``GMX_ENX_NO_FATAL``
        disable exiting upon encountering a corrupted frame in an :ref:`edr`
        file, allowing the use of all frames up until the corruption.

``GMX_FILLERS_IN_LOCAL_STATE``
        Fillers particles are needed to make the number of particles a multiple of the SIMD
        or GPU warp/wave-front width for computing non-bonded interactions. These fillers can
        also be added to the local state, if all algorithms support this, which avoids indexing
        but increases the size of buffers. This environment variable can be set to 0 or 2 to
        force this behavior off or on. Setting it to 1 turns the behavior on when supported.

``GMX_FORCE_UPDATE``
        update forces when invoking ``mdrun -rerun``.

``GMX_FORCE_GPU_AWARE_MPI``
        Override the result of build- and runtime GPU-aware MPI detection and force the use of
        direct GPU MPI communication. Aimed at cases where the user knows that the MPI library is
        GPU-aware, but |Gromacs| is not able to detect this. Note that only CUDA and SYCL builds
        support such functionality.

``GMX_FORCE_UPDATE_DEFAULT_CPU``
        Force update to run on the CPU by default, makes the ``mdrun -update auto`` behave as ``-update cpu``.

``GMX_GPU_DD_COMMS``
        Removed.

``GMX_GPU_DISABLE_COMPATIBILITY_CHECK``
        Disables the hardware compatibility check in OpenCL and SYCL. Useful for developers
        and allows testing the OpenCL/SYCL kernels on non-supported platforms without source code modification.

``GMX_GPU_ID``
        set in the same way as ``mdrun -gpu_id``, ``GMX_GPU_ID``
        allows the user to specify different GPU IDs for different ranks, which can be useful for selecting different
        devices on different compute nodes in a cluster.  Cannot be used in conjunction with ``mdrun -gpu_id``.

``GMX_GPU_NB_EWALD_TWINCUT``
        force the use of twin-range cutoff kernel even if :mdp:`rvdw` equals
        :mdp:`rcoulomb` after PP-PME load balancing. The switch to twin-range kernels is automated,
        so this variable should be used only for benchmarking.

``GMX_GPU_NB_ANA_EWALD``
        force the use of analytical Ewald kernels. Should be used only for benchmarking.

``GMX_GPU_NB_TAB_EWALD``
        force the use of tabulated Ewald kernels. Should be used only for benchmarking.

``GMX_GPU_PME_DECOMPOSITION``
        Enable the support for PME decomposition on GPU.
        This feature is supported with CUDA and SYCL backends, and allows
        using multiple PME ranks with GPU offload, which is expected to improve
        performance when scaling over many GPUs. Note: this feature still lacks
        substantial testing.

``GMX_GPU_DISABLE_BUFFER_OPS``
        Disable coordinate/force transformation to be done on the GPU and execute these
        tasks on the CPU instead.

``GMX_GPU_PME_PP_COMMS``
        Removed.

``GMX_GPUTASKS``
        set in the same way as ``mdrun -gputasks``, ``GMX_GPUTASKS`` allows the mapping
        of GPU tasks to GPU device IDs to be different on different ranks, if e.g. the MPI
        runtime permits this variable to be different for different ranks. Cannot be used
        in conjunction with ``mdrun -gputasks``. Has all the same requirements as ``mdrun -gputasks``.

.. _HeFFTe docs: https://icl-utk-edu.github.io/heffte/structheffte_1_1plan__options.html

``GMX_HEFFTE_RESHAPE_ALGORITHM``
        Sets ``heffte::plan_options::reshape_algorithm`` to ``p2p`` (the default) or ``p2p_plined``, ``alltoallv``, or ``alltoall``.
        See the `HeFFTe docs`_ for details.

``GMX_HEFFTE_USE_GPU_AWARE``
        Sets ``heffte::plan_options::use_gpu_aware`` to ``true`` (the default) or ``false``.
        See the `HeFFTe docs`_ for details.

``GMX_HEFFTE_USE_PENCILS``
        Sets ``heffte::plan_options::use_pencils`` to ``true`` or ``false`` (the default).
        See the `HeFFTe docs`_ for details.

``GMX_HEFFTE_USE_REORDER``
        Sets ``heffte::plan_options::use_reorder`` to ``true`` (the default) or ``false``.
        See the `HeFFTe docs`_ for details.

``GMX_IGNORE_FSYNC_FAILURE_ENV``
        allow :ref:`gmx mdrun` to continue even if
        a file is missing.

``GMX_LJCOMB_TOL``
        when set to a floating-point value, overrides the default tolerance of
        1e-5 for force-field floating-point parameters.

``GMX_MAXCONSTRWARN``
        if set to -1, :ref:`gmx mdrun` will
        not exit if it produces too many LINCS warnings.

``GMX_NB_MIN_CI``
        neighbor list balancing parameter used when running on GPU. Sets the
        target minimum number pair-lists in order to improve multi-processor load-balance for better
        performance with small simulation systems. Must be set to a non-negative integer,
        the 0 value disables list splitting.
        The default value is optimized for supported GPUs
        therefore changing it is not necessary for normal usage, but it can be useful on future architectures.

``GMX_NBNXN_CYCLE``
        when set, print detailed neighbor search cycle counting.

``GMX_NBNXN_EWALD_ANALYTICAL``
        force the use of analytical Ewald non-bonded kernels,
        mutually exclusive of ``GMX_NBNXN_EWALD_TABLE``.

``GMX_NBNXN_EWALD_TABLE``
        force the use of tabulated Ewald non-bonded kernels,
        mutually exclusive of ``GMX_NBNXN_EWALD_ANALYTICAL``.

``GMX_NBNXN_PLAINC_1X1``
        force the use of the reference 1x1 non-SIMD CPU non-bonded kernel,
        mutually exclusive of ``GMX_NBNXN_SIMD_2XNN`` and ``GMX_NBNXN_SIMD_4XN``.

``GMX_NBNXN_SIMD_2XNN``
        force the use of 2x(N+N) SIMD CPU non-bonded kernels,
        mutually exclusive of ``GMX_NBNXN_PLAINC_1X1`` and ``GMX_NBNXN_SIMD_4XN``.

``GMX_NBNXN_SIMD_4XN``
        force the use of 4xN SIMD CPU non-bonded kernels,
        mutually exclusive of ``GMX_NBNXN_PLAINC_1X1`` and ``GMX_NBNXN_SIMD_2XNN``.

``GMX_NO_CART_REORDER``
        used in initializing domain decomposition communicators. Rank reordering
        is default, but can be switched off with this environment variable.

``GMX_NO_INT``, ``GMX_NO_TERM``, ``GMX_NO_USR1``
        disable signal handlers for SIGINT,
        SIGTERM, and SIGUSR1, respectively.

``GMX_NO_LJ_COMB_RULE``
        force the use of LJ parameter lookup instead of using combination rules
        in the non-bonded kernels.

``GMX_NO_NODECOMM``
        do not use separate inter- and intra-node communicators.

``GMX_NO_NONBONDED``
        skip non-bonded calculations; can be used to estimate the possible
        performance gain from adding a GPU accelerator to the current hardware setup -- assuming that this is
        fast enough to complete the non-bonded calculations while the CPU does bonded force and PME computation.
        Freezing the particles will be required to stop the system blowing up.

``GMX_NO_UPDATEGROUPS``
        turns off update groups. May allow for a decomposition of more
        domains for small systems at the cost of communication during update.

``GMX_NOOPTIMIZEDKERNELS``
        deprecated, use ``GMX_DISABLE_SIMD_KERNELS`` instead.

``GMX_NOPREDICT``
        shell positions are not predicted.

``GMX_NSTLIST_DYNAMICPRUNING``
        overrides the dynamic pair-list pruning interval chosen heuristically
        by mdrun. Values should be between the pruning frequency value
        (1 for CPU and 2 for GPU) and :mdp:`nstlist` ``- 1``.

``GMX_PME_NUM_THREADS``
        set the number of OpenMP or PME threads; overrides the default set by
        :ref:`gmx mdrun`; can be used instead of the ``-npme`` command line option,
        also useful to set heterogeneous per-process/-node thread count.

``GMX_PME_P3M``
        use P3M-optimized influence function instead of smooth PME B-spline interpolation.

``GMX_PME_THREAD_DIVISION``
        PME thread division in the format "x y z" for all three dimensions. The
        sum of the threads in each dimension must equal the total number of PME threads (set in
        :envvar:`GMX_PME_NTHREADS`).

``GMX_PMEONEDD``
        if the number of domain decomposition cells is set to 1 for both x and y,
        decompose PME in one dimension.

``GMX_PULL_PARTICIPATE_ALL``
        disable the default heuristic for when to use a separate pull MPI communicator (at >=32 ranks).

``GMX_REQUIRE_SHELL_INIT``
        require that shell positions are initiated.

``GMX_TPIC_MASSES``
        should contain multiple masses used for test particle insertion into a cavity.
        The center of mass of the last atoms is used for insertion into the cavity.

``GMX_VERLET_BUFFER_PRESSURE_TOLERANCE``
        sets the maximum tolerated error in the pressure in bar for the
        automated tuning of the Verlet pair-list buffering. Can only be used
        with system where this tolerance has not been set using the mdp parameter.

``GMX_VERLET_BUFFER_RES``
        resolution of buffer size in Verlet cutoff scheme.  The default value is
        0.001, but can be overridden with this environment variable.

``HWLOC_XMLFILE``
        Not strictly a |Gromacs| environment variable, but on large machines
        the hwloc detection can take a few seconds if you have lots of MPI processes.
        If you run the hwloc command :command:`lstopo out.xml` and set this environment
        variable to point to the location of this file, the hwloc library will use
        the cached information instead, which can be faster.

``MDRUN``
        the :ref:`gmx mdrun` command used by :ref:`gmx tune_pme`.

``MPIRUN``
        the ``mpirun`` command used by :ref:`gmx tune_pme`.

.. _opencl-management:

OpenCL management
-----------------
Currently, several environment variables exist that help customize some aspects
of the OpenCL_ version of |Gromacs|. They are mostly related to the runtime
compilation of OpenCL kernels, but they are also used in device selection.

..
   Please keep these in alphabetical order!



``GMX_OCL_DEBUG``
        Use in conjunction with ``OCL_FORCE_CPU`` or with an AMD device.
        It adds the debug flag to the compiler options (-g).

``GMX_OCL_DISABLE_FASTMATH``
        Prevents the use of ``-cl-fast-relaxed-math`` compiler option.
        Note: fast math is always disabled on Intel devices due to instability.

``GMX_OCL_DISABLE_I_PREFETCH``
        Disables i-atom data (type or LJ parameter) prefetch allowing
        testing.

``GMX_OCL_ENABLE_I_PREFETCH``
        Enables i-atom data (type or LJ parameter) prefetch allowing
        testing on platforms where this behavior is not default.

``GMX_OCL_DUMP_INTERM_FILES``
        If defined, intermediate language code corresponding to the
        OpenCL build process is saved to file. Caching has to be
        turned off in order for this option to take effect.

            - NVIDIA GPUs: PTX code is saved in the current directory
              with the name ``device_name.ptx``
            - AMD GPUs: ``.IL/.ISA`` files will be created for each OpenCL
              kernel built.  For details about where these files are
              created check AMD documentation for ``-save-temps`` compiler
              option.

``GMX_OCL_DUMP_LOG``
        If defined, the OpenCL build log is always written to the
        mdrun log file. Otherwise, the build log is written to the
        log file only when an error occurs.

``GMX_OCL_FILE_PATH``
        Use this parameter to force |Gromacs| to load the OpenCL
        kernels from a custom location. Use it only if you want to
        override |Gromacs| default behavior, or if you want to test
        your own kernels.

``GMX_OCL_FORCE_AMD_WAVEFRONT64``
        Force the use of Wave64 mode on AMD devices. This allows using
        OpenCL on RDNA-family devices, but is not recommended.
        For development use only.

``GMX_OCL_FORCE_CPU``
        Force the selection of a CPU device instead of a GPU.  This
        exists only for debugging purposes. Do not expect |Gromacs| to
        function properly with this option on, it is solely for the
        simplicity of stepping in a kernel and see what is happening.

``GMX_OCL_GENCACHE``
        Enable OpenCL binary caching. Only intended to be used for
        development and (expert) testing as neither concurrency
        nor cache invalidation is implemented safely!

``GMX_OCL_NOFASTGEN``
        If set, generate and compile all algorithm flavors, otherwise
        only the flavor required for the simulation is generated and
        compiled.

``GMX_OCL_NOOPT``
        Disable optimisations. Adds the option ``cl-opt-disable`` to the
        compiler options.

``GMX_OCL_SHOW_DIAGNOSTICS``
        Use Intel OpenCL extension to show additional runtime performance
        diagnostics.

``GMX_OCL_VERBOSE``
        If defined, it enables verbose mode for OpenCL kernel build.
        Currently available only for NVIDIA GPUs. See ``GMX_OCL_DUMP_LOG``
        for details about how to obtain the OpenCL build log.

Analysis and Core Functions
---------------------------

..
   Please keep these in alphabetical order!

``GMX_DIPOLE_SPACING``
        spacing used by :ref:`gmx dipoles`.

``GMX_ENER_VERBOSE``
        make :ref:`gmx energy` and :ref:`gmx eneconv`
        loud and noisy.

``GMX_MAXRESRENUM``
        sets the maximum number of residues to be renumbered by
        :ref:`gmx grompp`. A value of -1 indicates all residues should be renumbered.

``GMX_NO_FFRTP_TER_RENAME``
        Some force fields (like AMBER) use specific names for N- and C-
        terminal residues (NXXX and CXXX) as :ref:`rtp` entries that are normally renamed. Setting
        this environment variable disables this renaming.

``GMX_USE_XMGR``
        sets viewer to ``xmgr`` (deprecated) instead of ``xmgrace``.

``GMXTIMEUNIT``
        the time unit used in output files, can be
        anything in fs, ps, ns, us, ms, s, m or h.


``VMD_PLUGIN_PATH``
        where to find VMD plug-ins. Needed to be
        able to read file formats recognized only by a VMD plug-in.

``VMDDIR``
        base path of VMD installation.
