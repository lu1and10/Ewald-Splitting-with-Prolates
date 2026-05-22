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
 */

/*! \internal \file
 * \brief
 * End-to-end mdrun smoke and accuracy tests for CPU ESP electrostatics.
 *
 * \ingroup module_mdrun_integration_tests
 */
#include "gmxpre.h"

#include <cmath>

#include <algorithm>
#include <filesystem>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "gromacs/trajectory/energyframe.h"
#include "gromacs/trajectory/trajectoryframe.h"
#include "gromacs/utility/stringutil.h"
#include "gromacs/utility/textwriter.h"
#include "gromacs/utility/vec.h"
#include "gromacs/utility/vectypes.h"

#include "testutils/cmdlinetest.h"
#include "testutils/simulationdatabase.h"
#include "testutils/trajectoryreader.h"

#include "energyreader.h"
#include "moduletest.h"

namespace gmx::test
{
namespace
{

struct RunOutput
{
    std::string trajectoryFileName;
    std::string energyFileName;
};

std::string makeEspMdp(const double accuracy, const int nsteps, const int nstfout, const int stencilOrder)
{
    return formatString("integrator               = md\n"
                        "nsteps                   = %d\n"
                        "dt                       = 0.001\n"
                        "continuation             = yes\n"
                        "constraints              = none\n"
                        "cutoff-scheme            = Verlet\n"
                        "coulombtype              = ESP\n"
                        "rcoulomb                 = 0.8\n"
                        "rlist                    = 0.8\n"
                        "vdwtype                  = Cut-off\n"
                        "rvdw                     = 0.8\n"
                        "esp-accuracy             = %.8g\n"
                        "esp-stencil-order        = %d\n"
                        "pbc                      = xyz\n"
                        "pcoupl                   = no\n"
                        "tcoupl                   = no\n"
                        "free-energy              = no\n"
                        "ewald-geometry           = 3d\n"
                        "nwall                    = 0\n"
                        "nstlist                  = 1\n"
                        "nstcalcenergy            = 1\n"
                        "nstenergy                = 1\n"
                        "nstxout                  = 0\n"
                        "nstvout                  = 0\n"
                        "nstfout                  = %d\n"
                        "nstlog                   = 0\n",
                        nsteps,
                        accuracy,
                        stencilOrder,
                        nstfout);
}

std::string makePmeMdp(const int nsteps,
                       const int nstfout,
                       const double fourierSpacing = 0.12,
                       const int    pmeOrder       = 4)
{
    return formatString("integrator               = md\n"
                        "nsteps                   = %d\n"
                        "dt                       = 0.001\n"
                        "continuation             = yes\n"
                        "constraints              = none\n"
                        "cutoff-scheme            = Verlet\n"
                        "coulombtype              = PME\n"
                        "rcoulomb                 = 0.8\n"
                        "rlist                    = 0.8\n"
                        "vdwtype                  = Cut-off\n"
                        "rvdw                     = 0.8\n"
                        "fourierspacing           = %.8g\n"
                        "pme-order                = %d\n"
                        "pbc                      = xyz\n"
                        "pcoupl                   = no\n"
                        "tcoupl                   = no\n"
                        "free-energy              = no\n"
                        "ewald-geometry           = 3d\n"
                        "nwall                    = 0\n"
                        "nstlist                  = 1\n"
                        "nstcalcenergy            = 1\n"
                        "nstenergy                = 1\n"
                        "nstxout                  = 0\n"
                        "nstvout                  = 0\n"
                        "nstfout                  = %d\n"
                        "nstlog                   = 0\n",
                        nsteps,
                        fourierSpacing,
                        pmeOrder,
                        nstfout);
}

CommandLine makeCpuMdrunCommandLine()
{
    CommandLine commandLine;
    commandLine.addOption("-reprod");
    commandLine.addOption("-notunepme");
    commandLine.addOption("-nb", "cpu");
    commandLine.addOption("-pme", "cpu");
    commandLine.addOption("-dd");
    commandLine.append("1");
    commandLine.append("1");
    commandLine.append("1");
    commandLine.addOption("-rdd", "0.3");
    return commandLine;
}

class EspIntegrationTest : public MdrunTestFixture
{
protected:
    void setUniqueOutputFileNames(SimulationRunner* runner, const std::string& label)
    {
        runner->fullPrecisionTrajectoryFileName_ =
                fileManager_.getTemporaryFilePath(label + ".trr").string();
        runner->groOutputFileName_ = fileManager_.getTemporaryFilePath(label + ".gro").string();
        runner->cptOutputFileName_ = fileManager_.getTemporaryFilePath(label + ".cpt").string();
        runner->mdpOutputFileName_ =
                fileManager_.getTemporaryFilePath(label + "-output.mdp").string();
        runner->tprFileName_ = fileManager_.getTemporaryFilePath(label + ".tpr").string();
        runner->logFileName_ = fileManager_.getTemporaryFilePath(label + ".log").string();
        runner->edrFileName_ = fileManager_.getTemporaryFilePath(label + ".edr").string();
        runner->mtxFileName_ = fileManager_.getTemporaryFilePath(label + ".mtx").string();
    }

    RunOutput runSpcWaterSystem(const std::string& label, const std::string& mdpContents)
    {
        SimulationRunner runner(&fileManager_);
        setUniqueOutputFileNames(&runner, label);
        runner.topFileName_ = fileManager_.getTemporaryFilePath(label + "-esp-spc216.top").string();
        runner.groFileName_ = TestFileManager::getInputFilePath("spc216.gro").string();
        runner.useStringAsMdpFile(mdpContents);

        TextWriter::writeFileFromString(
                runner.topFileName_,
                "[ defaults ]\n"
                "; nbfunc comb-rule gen-pairs fudgeLJ fudgeQQ\n"
                "1 2 yes 0.5 0.5\n"
                "\n"
                "[ atomtypes ]\n"
                "; name at.num mass charge ptype sigma epsilon\n"
                "OW 8 15.999 0.0 A 0.31506 0.6364\n"
                "HW 1 1.008  0.0 A 0.00000 0.0000\n"
                "\n"
                "[ moleculetype ]\n"
                "; name nrexcl\n"
                "SOL 2\n"
                "\n"
                "[ atoms ]\n"
                "; nr type resnr residue atom cgnr charge mass\n"
                "1 OW 1 SOL OW  1 -0.834 15.999\n"
                "2 HW 1 SOL HW1 1  0.417 1.008\n"
                "3 HW 1 SOL HW2 1  0.417 1.008\n"
                "\n"
                "[ bonds ]\n"
                "1 2 1 0.1 345000\n"
                "1 3 1 0.1 345000\n"
                "\n"
                "[ angles ]\n"
                "2 1 3 1 109.47 383\n"
                "\n"
                "[ system ]\n"
                "ESP integration water\n"
                "\n"
                "[ molecules ]\n"
                "SOL 216\n");

        if (runner.callGrompp() != 0)
        {
            ADD_FAILURE() << "grompp failed for minimal water system";
            return {};
        }
        CommandLine commandLine = makeCpuMdrunCommandLine();
        if (runner.callMdrun(commandLine) != 0)
        {
            ADD_FAILURE() << "mdrun failed for minimal water system";
            return {};
        }
        return { runner.fullPrecisionTrajectoryFileName_, runner.edrFileName_ };
    }

    RunOutput runDatabaseSystem(const std::string& systemName,
                                const std::string& mdpContents,
                                const int          maxWarnings = 0)
    {
        SimulationRunner runner(&fileManager_);
        runner.useTopGroAndNdxFromDatabase(systemName);
        if (!std::filesystem::exists(runner.ndxFileName_))
        {
            runner.ndxFileName_.clear();
        }
        runner.useStringAsMdpFile(mdpContents);
        runner.setMaxWarn(maxWarnings);
        if (runner.callGrompp() != 0)
        {
            ADD_FAILURE() << "grompp failed for " << systemName;
            return {};
        }
        CommandLine commandLine = makeCpuMdrunCommandLine();
        if (runner.callMdrun(commandLine) != 0)
        {
            ADD_FAILURE() << "mdrun failed for " << systemName;
            return {};
        }
        return { runner.fullPrecisionTrajectoryFileName_, runner.edrFileName_ };
    }

    RunOutput runInlineIonSystem(const std::string& mdpContents)
    {
        SimulationRunner runner(&fileManager_);
        runner.topFileName_ = fileManager_.getTemporaryFilePath("esp-ions.top").string();
        runner.groFileName_ = fileManager_.getTemporaryFilePath("esp-ions.gro").string();
        runner.useStringAsMdpFile(mdpContents);
        runner.setMaxWarn(1);

        TextWriter::writeFileFromString(
                runner.topFileName_,
                "[ defaults ]\n"
                "; nbfunc comb-rule gen-pairs fudgeLJ fudgeQQ\n"
                "1 2 yes 0.5 0.5\n"
                "\n"
                "[ atomtypes ]\n"
                "; name at.num mass charge ptype sigma epsilon\n"
                "NA 11 22.9898 0.0 A 0.257 0.1\n"
                "CL 17 35.4500 0.0 A 0.440 0.1\n"
                "\n"
                "[ moleculetype ]\n"
                "NA 1\n"
                "\n"
                "[ atoms ]\n"
                "1 NA 1 NA NA 1 1.0 22.9898\n"
                "\n"
                "[ moleculetype ]\n"
                "CL 1\n"
                "\n"
                "[ atoms ]\n"
                "1 CL 1 CL CL 1 -1.0 35.45\n"
                "\n"
                "[ system ]\n"
                "ESP non-neutral ion smoke\n"
                "\n"
                "[ molecules ]\n"
                "NA 2\n"
                "CL 1\n");

        TextWriter::writeFileFromString(runner.groFileName_,
                                        "ESP non-neutral ion smoke\n"
                                        "3\n"
                                        "    1NA      NA    1   0.500   0.500   0.500\n"
                                        "    2NA      NA    2   1.500   1.500   1.500\n"
                                        "    3CL      CL    3   2.500   2.500   2.500\n"
                                        "   3.00000   3.00000   3.00000\n");

        if (runner.callGrompp() != 0)
        {
            ADD_FAILURE() << "grompp failed for inline non-neutral ion system";
            return {};
        }
        CommandLine commandLine = makeCpuMdrunCommandLine();
        if (runner.callMdrun(commandLine) != 0)
        {
            ADD_FAILURE() << "mdrun failed for inline non-neutral ion system";
            return {};
        }
        return { runner.fullPrecisionTrajectoryFileName_, runner.edrFileName_ };
    }
};

std::vector<RVec> readLastForces(const std::string& trajectoryFileName)
{
    std::vector<RVec> forces;
    TrajectoryFrameReader reader(trajectoryFileName);
    while (reader.readNextFrame())
    {
        const auto frame = reader.frame();
        forces.assign(frame.f().begin(), frame.f().end());
    }
    if (forces.empty())
    {
        ADD_FAILURE() << "No force frames found in " << trajectoryFileName;
    }
    return forces;
}

real readLastElectrostaticEnergy(const std::string& energyFileName)
{
    constexpr const char* c_coulombShortRange = "Coulomb (SR)";
    constexpr const char* c_coulombReciprocal = "Coul. recip.";

    bool foundFrame = false;
    real energy     = 0;
    auto reader =
            openEnergyFileToReadTerms(energyFileName, { c_coulombShortRange, c_coulombReciprocal });
    while (reader->readNextFrame())
    {
        const EnergyFrame& frame = reader->frame();
        energy = frame.at(c_coulombShortRange) + frame.at(c_coulombReciprocal);
        foundFrame = true;
    }
    if (!foundFrame)
    {
        ADD_FAILURE() << "No energy frames found in " << energyFileName;
    }
    return energy;
}

double relativeL2ForceError(const std::vector<RVec>& testForces, const std::vector<RVec>& refForces)
{
    EXPECT_EQ(testForces.size(), refForces.size());
    if (testForces.size() != refForces.size() || testForces.empty())
    {
        return 0.0;
    }

    double numerator   = 0;
    double denominator = 0;
    for (size_t atom = 0; atom < testForces.size(); ++atom)
    {
        for (int dim = 0; dim < DIM; ++dim)
        {
            const double diff = testForces[atom][dim] - refForces[atom][dim];
            numerator += diff * diff;
            denominator += static_cast<double>(refForces[atom][dim]) * refForces[atom][dim];
        }
    }
    return std::sqrt(numerator / std::max(denominator, 1.0e-30));
}

double relativeScalarError(const real testValue, const real refValue)
{
    return std::abs(static_cast<double>(testValue) - refValue)
           / std::max(std::abs(static_cast<double>(refValue)), 1.0e-30);
}

void expectFiniteForces(const std::vector<RVec>& forces)
{
    ASSERT_FALSE(forces.empty());
    for (size_t atom = 0; atom < forces.size(); ++atom)
    {
        for (int dim = 0; dim < DIM; ++dim)
        {
            EXPECT_TRUE(std::isfinite(forces[atom][dim])) << "atom=" << atom << " dim=" << dim;
        }
    }
}

TEST_F(EspIntegrationTest, SpcEWater_ForceAndEnergyErrorVsHighAccuracyPme)
{
    const RunOutput reference =
            runSpcWaterSystem("spce-high-accuracy-pme", makePmeMdp(1, 1, 0.04, 6));
    const RunOutput test = runSpcWaterSystem("spce-esp", makeEspMdp(1.0e-4, 1, 1, -1));

    const std::vector<RVec> refForces  = readLastForces(reference.trajectoryFileName);
    const std::vector<RVec> testForces = readLastForces(test.trajectoryFileName);
    expectFiniteForces(refForces);
    expectFiniteForces(testForces);

    const double delta = relativeL2ForceError(testForces, refForces);
    EXPECT_TRUE(std::isfinite(delta));
    EXPECT_LT(delta, 0.1) << "force delta=" << delta;

    const real   refEnergy   = readLastElectrostaticEnergy(reference.energyFileName);
    const real   testEnergy  = readLastElectrostaticEnergy(test.energyFileName);
    const double energyDelta = relativeScalarError(testEnergy, refEnergy);
    EXPECT_TRUE(std::isfinite(refEnergy));
    EXPECT_TRUE(std::isfinite(testEnergy));
    EXPECT_TRUE(std::isfinite(energyDelta));
    EXPECT_LT(energyDelta, 0.1) << "energy delta=" << energyDelta << " reference=" << refEnergy
                                << " test=" << testEnergy;
}

TEST_F(EspIntegrationTest, SpcEWater_AgreementWithPme)
{
    const RunOutput esp = runSpcWaterSystem("agreement-esp", makeEspMdp(1.0e-4, 1, 1, -1));
    const RunOutput pme = runSpcWaterSystem("agreement-pme", makePmeMdp(1, 1));

    const std::vector<RVec> espForces = readLastForces(esp.trajectoryFileName);
    const std::vector<RVec> pmeForces = readLastForces(pme.trajectoryFileName);
    expectFiniteForces(espForces);
    expectFiniteForces(pmeForces);

    const double delta = relativeL2ForceError(espForces, pmeForces);
    EXPECT_TRUE(std::isfinite(delta));
    EXPECT_LT(delta, 2.0);
}

TEST_F(EspIntegrationTest, NaClNonNeutral_HighQ2SumSmoke)
{
    const RunOutput output = runInlineIonSystem(makeEspMdp(1.0e-4, 1, 1, -1));

    const std::vector<RVec> forces = readLastForces(output.trajectoryFileName);
    ASSERT_EQ(forces.size(), 3U);
    expectFiniteForces(forces);
}

TEST_F(EspIntegrationTest, Lysozyme_SingleStepSmoke)
{
    const RunOutput output = runDatabaseSystem("lysozyme", makeEspMdp(1.0e-4, 1, 1, -1), 1);

    const std::vector<RVec> forces = readLastForces(output.trajectoryFileName);
    ASSERT_EQ(forces.size(), 156U);
    expectFiniteForces(forces);
}

} // namespace
} // namespace gmx::test
