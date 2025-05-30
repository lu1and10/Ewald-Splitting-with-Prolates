#
# This file is part of the GROMACS molecular simulation package.
#
# Copyright 2012- The GROMACS Authors
# and the project initiators Erik Lindahl, Berk Hess and David van der Spoel.
# Consult the AUTHORS/COPYING files and https://www.gromacs.org for details.
#
# GROMACS is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public License
# as published by the Free Software Foundation; either version 2.1
# of the License, or (at your option) any later version.
#
# GROMACS is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with GROMACS; if not, see
# https://www.gnu.org/licenses, or write to the Free Software Foundation,
# Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA.
#
# If you want to redistribute modifications to GROMACS, please
# consider that scientific software is very special. Version
# control is crucial - bugs must be traceable. We will be happy to
# consider code for inclusion in the official distribution, but
# derived work must not be called official GROMACS. Details are found
# in the README & COPYING files - if they are missing, get the
# official version at https://www.gromacs.org.
#
# To help us fund GROMACS development, we humbly ask that you cite
# the research papers on the package. Check out https://www.gromacs.org.

# Manage the GROMACS shared library setup.

########################################################################
# Shared/static library settings
########################################################################
# Determine the defaults (this block has no effect if the variables have
# already been set)
if((APPLE OR CYGWIN OR ${CMAKE_SYSTEM_NAME} MATCHES "Linux|.*BSD|GNU"))
    # Maybe Solaris should be here? Patch this if you know!
    set(SHARED_LIBS_DEFAULT ON)
elseif(WIN32)
    # Support for shared libs on native Windows is a bit new. Its
    # default might change later if/when we sort things out. Also,
    # Cray should go here. What variable value can detect it?
    set(SHARED_LIBS_DEFAULT OFF)
else()
    if (NOT DEFINED BUILD_SHARED_LIBS)
        message(STATUS "Defaulting to building static libraries")
    endif()
    set(SHARED_LIBS_DEFAULT OFF)
endif()

if (GMX_PREFER_STATIC_LIBS)
    if (NOT DEFINED BUILD_SHARED_LIBS AND SHARED_LIBS_DEFAULT)
        message("Searching for static libraries requested, so the GROMACS libraries will also be static (BUILD_SHARED_LIBS=OFF)")
    endif()
    set(SHARED_LIBS_DEFAULT OFF)
endif()

if (NOT GMX_BUILD_SHARED_EXE)
    set(GMX_PREFER_STATIC_LIBS_DEFAULT ON)
    set(SHARED_LIBS_DEFAULT OFF)
endif()

# Declare the user-visible options
option(BUILD_SHARED_LIBS "Enable shared libraries (can be problematic e.g. with MPI, or on some HPC systems)" ${SHARED_LIBS_DEFAULT})

set(GMX_PREFER_STATIC_LIBS_DEFAULT OFF)
if (WIN32 AND NOT BUILD_SHARED_LIBS)
    set(GMX_PREFER_STATIC_LIBS_DEFAULT ON)
endif()

if (UNIX)
    set(GMX_PREFER_STATIC_LIBS_DESCRIPTION
        "When finding libraries prefer static archives (it will only work if static versions of external dependencies are available and found)")
elseif (WIN32)
    set(GMX_PREFER_STATIC_LIBS_DESCRIPTION
        "When finding libraries prefer static system libraries (MT instead of MD)")
endif()
option(GMX_PREFER_STATIC_LIBS "${GMX_PREFER_STATIC_LIBS_DESCRIPTION}"
       ${GMX_PREFER_STATIC_LIBS_DEFAULT})
mark_as_advanced(GMX_PREFER_STATIC_LIBS)

# Act on the set values
if (UNIX AND GMX_PREFER_STATIC_LIBS)
    if(BUILD_SHARED_LIBS)
        # Warn the user about the combination. But don't overwrite the request.
        message(WARNING "Searching for static libraries requested, and building shared GROMACS libraries requested. This might cause problems linking later.")
    endif()
    # On Linux .a is the static library suffix, on Mac OS X .lib can also
    # be used, so we'll add both to the preference list.
    set(CMAKE_FIND_LIBRARY_SUFFIXES ".lib;.a" ${CMAKE_FIND_LIBRARY_SUFFIXES})
endif()

# ==========
# Only things for managing shared libraries and build types on Windows follow

# Change the real CMake variables so we prefer static linking. This
# should be a function so we can have proper local variables while
# avoiding duplicating code.
function(gmx_manage_prefer_static_libs_flags build_type)
    if("${build_type}" STREQUAL "")
        set(punctuation "") # for general compiler flags (e.g.) CMAKE_CXX_FLAGS
    else()
        set(punctuation "_") # for build-type-specific compiler flags (e.g.) CMAKE_CXX_FLAGS_RELEASE
    endif()

    # Change the real CMake variables for the given build type in each
    # language, in the parent scope.
    foreach(language C CXX)
        set(varname "CMAKE_${language}_FLAGS${punctuation}${build_type}")
        string(REPLACE /MD /MT ${varname} "${${varname}}")
        set(${varname} "${${varname}}" PARENT_SCOPE)
    endforeach()
endfunction()

if( WIN32)
  if (NOT BUILD_SHARED_LIBS)
      if(NOT GMX_PREFER_STATIC_LIBS)
          message(WARNING "Shared system libraries requested, and static GROMACS libraries requested.")
      endif()
  else()
      if(MINGW)
          set(CMAKE_SHARED_LINKER_FLAGS "-Wl,--export-all-symbols ${CMAKE_SHARED_LINKER_FLAGS}")
      else()
          set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
      endif()
      if(GMX_PREFER_STATIC_LIBS)
          #this combination segfaults (illegal passing of file handles)
          message(FATAL_ERROR "Static system libraries requested, and shared GROMACS libraries requested.")
      endif()
      # Visibility not yet implemented
      # add_definitions(-DUSE_VISIBILITY -DTMPI_USE_VISIBILITY)
      # set(PKG_CFLAGS "$PKG_CFLAGS -DUSE_VISIBILITY -DTMPI_USE_VISIBILITY")
  endif()

  if(GMX_PREFER_STATIC_LIBS)
      foreach(build_type "" ${build_types_with_explicit_flags})
          gmx_manage_prefer_static_libs_flags("${build_type}")
      endforeach()
  endif()
  if( CMAKE_C_COMPILER_ID STREQUAL "Intel" )
    if(BUILD_SHARED_LIBS) #not sure why incremental building with shared libs doesn't work
        string(REPLACE "/INCREMENTAL:YES" "" CMAKE_SHARED_LINKER_FLAGS ${CMAKE_SHARED_LINKER_FLAGS})
        set(CMAKE_SHARED_LINKER_FLAGS ${CMAKE_SHARED_LINKER_FLAGS} PARENT_SCOPE)
    endif()
  endif()
endif()
