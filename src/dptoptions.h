#pragma once

#include "commondef.h"

#include <string>

struct DptOptions {
    std::string integrator = "mc";  // MC or MCMC
    bool bidirectional = true;
    int spp = 16;
    int numInitSamples = 100000;
    Float largeStepProb = Float(0.3);
    int minDepth = -1;
    int maxDepth = -1;
    int directSpp = 16;
    bool h2mc = true;
    Float perturbStdDev = Float(0.01);
    Float roughnessThreshold = Float(0.03);
    Float lensPerturbProb = Float(0.3);
    Float lensPerturbStdDev = Float(0.01);
    int numChains = 1024;
    int seedOffset = 0;
    int reportIntervalSpp = 0;
    bool useLightCoordinateSampling = true;
    Float discreteStdDev = Float(0.01);
    bool largeStepMultiplexed = true;
    Float uniformMixingProbability = Float(0.2);
};

inline std::string GetLibPath() {
    return std::string(getenv("DPT_LIBPATH"));
}
