#include "mlt.h"
#include "scene.h"
#include "path.h"
#include "camera.h"
#include "progressreporter.h"
#include "parallel.h"
#include "timer.h"
#include "h2mc.h"
#include "gaussian.h"
#include "alignedallocator.h"
#include "distribution.h"

#include <vector>
#include <mutex>

/**
 *  We implement a hybrid algorithm that combines Primary Sample Space MLT [Kelemen et al. 2002]
 *  and Multiplxed MLT (MMLT) [Hachisuka et al. 2014].  Specifically, the state of our Markov
 *  chain only represents one of the N^2 pairs connection as in MMLT.  During the "large
 *  step" mutations, instead of choosing the camera and light subpath lengthes a priori as in
 *  MMLT, we sample all pairs of connections, and probabilistically pick one based on their
 *  contributions (similar to Multiple-try Metropolis).  During the "small step" mutations,
 *  we fix the camera and light subpath lengthes of the state.
 */


static void DirectLighting(const Scene *scene, SampleBuffer &buffer) {
    if (scene->options->minDepth > 2 || scene->options->maxDepth < 1) {
        return;
    }

    std::cout << "Compute direct lighting" << std::endl;
    const Camera *camera = scene->camera.get();
    const int pixelHeight = GetPixelHeight(camera);
    const int pixelWidth = GetPixelWidth(camera);
    const int tileSize = 16;
    const int nXTiles = (pixelWidth + tileSize - 1) / tileSize;
    const int nYTiles = (pixelHeight + tileSize - 1) / tileSize;
    ProgressReporter reporter(nXTiles * nYTiles);

    Timer timer;
    Tick(timer);
    ParallelFor([&](const Vector2i tile) {
        const int seed = tile[1] * nXTiles + tile[0] + scene->options->seedOffset;
        RNG rng(seed);
        const int x0 = tile[0] * tileSize;
        const int x1 = std::min(x0 + tileSize, pixelWidth);
        const int y0 = tile[1] * tileSize;
        const int y1 = std::min(y0 + tileSize, pixelHeight);
        Path path;
        for (int y = y0; y < y1; y++) {
            for (int x = x0; x < x1; x++) {
                for (int s = 0; s < scene->options->directSpp; s++) {
                    std::vector<SubpathContrib> spContribs;
                    Clear(path);
                    GeneratePath(scene,
                                 Vector2i(x, y),
                                 std::min(scene->options->minDepth, 2),
                                 std::min(scene->options->maxDepth, 2),
                                 path,
                                 spContribs,
                                 rng);
                    for (const auto &spContrib : spContribs) {
                        Vector3 contrib = spContrib.contrib;
                        Splat(buffer, spContrib.screenPos, contrib);
                    }
                }
            }
        }
        reporter.Update(1);
    }, Vector2i(nXTiles, nYTiles));
    TerminateWorkerThreads();
    reporter.Done();
    Float elapsed = Tick(timer);
    std::cout << "Elapsed time:" << elapsed << std::endl;
}

struct MLTState {
    const Scene *scene;
    const decltype(&GeneratePath) genPathFunc;
    const decltype(&PerturbPath) perturbPathFunc;
    const decltype(&PerturbLens) perturbLensFunc;
    PathFuncMap funcMap;
    PathFuncDervMap funcDervMap;
    PathFuncMap staticFuncMap;
    PathFuncDervMap staticFuncDervMap;
    PathFuncMap lensFuncMap;
    PathFuncDervMap lensFuncDervMap;
};

struct SplatSample {
    Vector2 screenPos;
    Vector3 contrib;
};

struct MarkovState {
    bool valid;
    SubpathContrib spContrib;
    Path path;
    Float scoreSum;

    bool gaussianInitialized;
    Gaussian gaussian;
    bool lensGaussianInitialized;
    Gaussian lensGaussian;

    std::vector<SplatSample> toSplat;
};

static Float MLTInit(const MLTState &mltState,
                     const int64_t numInitSamples,
                     const int numChains,
                     std::vector<MarkovState> &initStates,
                     std::shared_ptr<PiecewiseConstant1D> &lengthDist) {
    std::cout << "Initializing mlt" << std::endl;
    Timer timer;
    Tick(timer);

    const int64_t numSamplesPerThread = numInitSamples / NumSystemCores();
    const int64_t threadsNeedExtraSamples = numSamplesPerThread % NumSystemCores();
    const Scene *scene = mltState.scene;
    auto genPathFunc = mltState.genPathFunc;

    std::mutex mStateMutex;
    struct LightMarkovState {
        RNG rng;
        int camDepth;
        int lightDepth;
        Float lsScore;
    };
    std::vector<LightMarkovState> mStates;
    Float totalScore(Float(0.0));
    std::vector<Float> lengthContrib;
    ParallelFor([&](const int threadId) {
        RNG rng(threadId + scene->options->seedOffset);
        int64_t numSamplesThisThread =
            numSamplesPerThread + ((threadIndex < threadsNeedExtraSamples) ? 1 : 0);
        std::vector<SubpathContrib> spContribs;
        Path path;
        for (int sampleIdx = 0; sampleIdx < numSamplesThisThread; sampleIdx++) {
            spContribs.clear();
            RNG rngCheckpoint = rng;
            Clear(path);
            const int minPathLength = std::max(scene->options->minDepth, 3);
            genPathFunc(scene,
                        Vector2i(-1, -1),
                        minPathLength,
                        scene->options->maxDepth,
                        path,
                        spContribs,
                        rng);

            std::lock_guard<std::mutex> lock(mStateMutex);
            for (const auto &spContrib : spContribs) {
                totalScore += spContrib.lsScore;
                const int pathLength = GetPathLength(spContrib.camDepth, spContrib.lightDepth);
                if (pathLength >= int(lengthContrib.size())) {
                    lengthContrib.resize(pathLength + 1, Float(0.0));
                }
                lengthContrib[pathLength] += spContrib.lsScore;
                mStates.emplace_back(LightMarkovState{
                    rngCheckpoint, spContrib.camDepth, spContrib.lightDepth, spContrib.lsScore});
            }
        }
    }, NumSystemCores());

    lengthDist = std::make_shared<PiecewiseConstant1D>(&lengthContrib[0], lengthContrib.size());

    if (int(mStates.size()) < numChains) {
        Error(
            "MLT initialization failed, consider using a larger number of initial samples or "
            "smaller number of chains");
    }

    // Equal-spaced seeding (See p.340 in Veach's thesis)
    std::vector<Float> cdf(mStates.size() + 1);
    cdf[0] = Float(0.0);
    for (int i = 0; i < (int)mStates.size(); i++) {
        cdf[i + 1] = cdf[i] + mStates[i].lsScore;
    }
    const Float interval = cdf.back() / Float(numChains);
    std::uniform_real_distribution<Float> uniDist(Float(0.0), interval);
    RNG rng(mStates.size());
    Float pos = uniDist(rng);
    int cdfPos = 0;
    initStates.reserve(numChains);
    std::vector<SubpathContrib> spContribs;
    for (int i = 0; i < (int)numChains; i++) {
        while (pos > cdf[cdfPos]) {
            cdfPos = std::min(cdfPos + 1, mStates.size() - 1);
        }
        initStates.push_back(MarkovState{true});
        MarkovState &state = initStates.back();
        spContribs.clear();
        Clear(state.path);
        genPathFunc(scene,
                    Vector2i(-1, -1),
                    std::max(scene->options->minDepth, 3),
                    scene->options->maxDepth,
                    state.path,
                    spContribs,
                    mStates[cdfPos - 1].rng);
        state.scoreSum = Float(0.0);
        for (const auto &spContrib : spContribs) {
            state.scoreSum += spContrib.lsScore;
            if (spContrib.camDepth == mStates[cdfPos - 1].camDepth &&
                spContrib.lightDepth == mStates[cdfPos - 1].lightDepth) {
                state.spContrib = spContrib;
            }
        }
        ToSubpath(state.spContrib.camDepth, state.spContrib.lightDepth, state.path);
        state.gaussianInitialized = false;
        state.lensGaussianInitialized = false;
        pos += interval;
    }

    Float invNumInitSamples = inverse(Float(numInitSamples));
    Float elapsed = Tick(timer);
    std::cout << "Elapsed time:" << elapsed << std::endl;
    return Float(totalScore) * invNumInitSamples;
}

enum class MutationType { Large, Small, Lens, H2MCSmall, H2MCLens };

struct Mutation {
    virtual Float Mutate(const MLTState &mltState,
                         const Float normalization,
                         MarkovState &currentState,
                         MarkovState &proposalState,
                         RNG &rng) = 0;

    MutationType lastMutationType;
};

struct LargeStep : public Mutation {
    LargeStep(std::shared_ptr<PiecewiseConstant1D> lengthDist) : lengthDist(lengthDist) {
    }

    Float Mutate(const MLTState &mltState,
                 const Float normalization,
                 MarkovState &currentState,
                 MarkovState &proposalState,
                 RNG &rng) override;

    std::shared_ptr<PiecewiseConstant1D> lengthDist;
    std::vector<SubpathContrib> spContribs;
    std::vector<Float> contribCdf;
    Float lastScoreSum = Float(1.0);
    Float lastScore = Float(1.0);
};

Float LargeStep::Mutate(const MLTState &mltState,
                        const Float normalization,
                        MarkovState &currentState,
                        MarkovState &proposalState,
                        RNG &rng) {
    lastMutationType = MutationType::Large;
    std::uniform_real_distribution<Float> uniDist(Float(0.0), Float(1.0));
    const Scene *scene = mltState.scene;
    const auto genPathFunc = mltState.genPathFunc;
    Float a = Float(1.0);
    spContribs.clear();
    Clear(proposalState.path);
    if (scene->options->largeStepMultiplexed) {
        int length = lengthDist->SampleDiscrete(uniDist(rng), nullptr);
        int lgtLength = scene->options->bidirectional
                            ? Clamp(int(uniDist(rng) * (length + 1)), 0, length)
                            : Clamp(int(uniDist(rng) * 2), 0, 1);
        int camLength = length - lgtLength + 1;
        GenerateSubpath(scene,
                        Vector2i(-1, -1),
                        camLength,
                        lgtLength,
                        scene->options->bidirectional,
                        proposalState.path,
                        spContribs,
                        rng);
        assert(spContribs.size() <= 1);
    } else {
        genPathFunc(scene,
                    Vector2i(-1, -1),
                    std::max(scene->options->minDepth, 3),
                    scene->options->maxDepth,
                    proposalState.path,
                    spContribs,
                    rng);
    }
    proposalState.gaussianInitialized = false;
    proposalState.lensGaussianInitialized = false;
    if (spContribs.size() > 0) {
        contribCdf.clear();
        contribCdf.push_back(Float(0.0));
        for (const auto &spContrib : spContribs) {
            contribCdf.push_back(contribCdf.back() + spContrib.lsScore);
        }
        const Float scoreSum = contribCdf.back();
        const Float invSc = inverse(scoreSum);
        std::for_each(contribCdf.begin(), contribCdf.end(), [invSc](Float &cdf) { cdf *= invSc; });

        const auto it = std::upper_bound(contribCdf.begin(), contribCdf.end(), uniDist(rng));
        int64_t contribId =
            Clamp(int64_t(it - contribCdf.begin() - 1), int64_t(0), int64_t(spContribs.size() - 1));
        proposalState.spContrib = spContribs[contribId];
        proposalState.scoreSum = scoreSum;

        if (currentState.valid) {
            if (scene->options->largeStepMultiplexed) {
                int currentLength = GetPathLength(currentState.spContrib.camDepth,
                                                  currentState.spContrib.lightDepth);
                int proposalLength = GetPathLength(proposalState.spContrib.camDepth,
                                                   proposalState.spContrib.lightDepth);
                Float invProposalTechniquesPmf = scene->options->bidirectional
                                                     ? (Float(proposalLength) + Float(1.0))
                                                     : Float(2.0);
                Float invCurrentTechniquesPmf = scene->options->bidirectional
                                                    ? (Float(currentLength) + Float(1.0))
                                                    : Float(2.0);
                a = Clamp((invProposalTechniquesPmf * proposalState.spContrib.lsScore /
                           lengthDist->Pmf(proposalLength)) /
                              (invCurrentTechniquesPmf * currentState.spContrib.lsScore /
                               lengthDist->Pmf(currentLength)),
                          Float(0.0),
                          Float(1.0));
            } else {
                // In general, we do not have the "scoreSum" of currentState, since small steps only
                // mutate one subpath
                // To address this, we introduce an augmented space that only contains large step
                // states.
                const Float probProposal =
                    (proposalState.spContrib.lsScore / proposalState.scoreSum);
                const Float probLast = (lastScore / lastScoreSum);
                a = Clamp((proposalState.spContrib.lsScore * probLast) /
                              (currentState.spContrib.lsScore * probProposal),
                          Float(0.0),
                          Float(1.0));
            }
        }

        proposalState.toSplat.clear();
        // proposalState.toSplat.push_back(SplatSample{
        //    proposalState.spContrib.screenPos,
        //    proposalState.spContrib.contrib * (normalization / proposalState.spContrib.lsScore)});
        for (const auto &spContrib : spContribs) {
            proposalState.toSplat.push_back(
                SplatSample{spContrib.screenPos, spContrib.contrib * (normalization / scoreSum)});
        }
    } else {
        a = Float(0.0);
    }
    return a;
}


struct SmallStep : public Mutation {
    Float Mutate(const MLTState &mltState,
                 const Float normalization,
                 MarkovState &currentState,
                 MarkovState &proposalState,
                 RNG &rng) override;

    std::vector<SubpathContrib> spContribs;
};

Float SmallStep::Mutate(const MLTState &mltState,
                        const Float normalization,
                        MarkovState &currentState,
                        MarkovState &proposalState,
                        RNG &rng) {
    const Scene *scene = mltState.scene;
    spContribs.clear();

    Float a = Float(1.0);
    assert(currentState.valid);
    proposalState.path = currentState.path;
    const Float lensPerturbProb = scene->options->lensPerturbProb;
    std::uniform_real_distribution<Float> uniDist(Float(0.0), Float(1.0));
    if (currentState.spContrib.lensScore > Float(0.0) && uniDist(rng) < lensPerturbProb) {
        const Float stdDev = scene->options->lensPerturbStdDev;
        std::normal_distribution<Float> normDist(Float(0.0), stdDev);
        lastMutationType = MutationType::Lens;
        const auto perturbLensFunc = mltState.perturbLensFunc;
        Vector2 offset;
        offset[0] = normDist(rng);
        offset[1] = normDist(rng);
        Vector2 screenPos = currentState.spContrib.screenPos;
        screenPos[0] = Modulo(screenPos[0] + offset[0], Float(1.0));
        screenPos[1] = Modulo(screenPos[1] + offset[1], Float(1.0));
        perturbLensFunc(scene, screenPos, proposalState.path, spContribs);
        SetIsMoving(proposalState.path);
        proposalState.gaussianInitialized = false;
        proposalState.lensGaussianInitialized = false;
        if (spContribs.size() > 0) {
            proposalState.spContrib = spContribs[0];
            a = Clamp(proposalState.spContrib.lensScore / currentState.spContrib.lensScore,
                      Float(0.0),
                      Float(1.0));
            proposalState.toSplat.clear();
            for (const auto &spContrib : spContribs) {
                proposalState.toSplat.push_back(SplatSample{
                    spContrib.screenPos, spContrib.contrib * (normalization / spContrib.lsScore)});
            }
        } else {
            a = Float(0.0);
        }
    } else {
        const Float stdDev = scene->options->perturbStdDev;
        std::normal_distribution<Float> normDist(Float(0.0), stdDev);
        lastMutationType = MutationType::Small;
        const auto perturbPathFunc = mltState.perturbPathFunc;
        Vector offset(GetDimension(currentState.path));
        for (int i = 0; i < offset.size(); i++) {
            offset[i] = normDist(rng);
        }
        perturbPathFunc(scene, offset, proposalState.path, spContribs, rng);
        SetIsMoving(proposalState.path);
        proposalState.gaussianInitialized = false;
        proposalState.lensGaussianInitialized = false;
        if (spContribs.size() > 0) {
            assert(spContribs.size() == 1);
            proposalState.spContrib = spContribs[0];
            a = Clamp(proposalState.spContrib.ssScore / currentState.spContrib.ssScore,
                      Float(0.0),
                      Float(1.0));
            proposalState.toSplat.clear();
            for (const auto &spContrib : spContribs) {
                proposalState.toSplat.push_back(SplatSample{
                    spContrib.screenPos, spContrib.contrib * (normalization / spContrib.lsScore)});
            }
        } else {
            a = Float(0.0);
        }
    }
    return a;
}

using DervFuncMap = std::unordered_map<std::pair<int, int>, PathFuncDerv>;

struct H2MCSmallStep : public Mutation {
    H2MCSmallStep(const Scene *scene,
                  const int maxDervDepth,
                  const Float sigma,
                  const Float lensSigma);
    Float Mutate(const MLTState &mltState,
                 const Float normalization,
                 MarkovState &currentState,
                 MarkovState &proposalState,
                 RNG &rng) override;

    std::vector<SubpathContrib> spContribs;
    H2MCParam h2mcParam, lensH2mcParam;
    AlignedStdVector sceneParams;
    SerializedSubpath ssubPath;
    SmallStep isotropicSmallStep;

    AlignedStdVector vGrad;
    AlignedStdVector vHess;
};

H2MCSmallStep::H2MCSmallStep(const Scene *scene,
                             const int maxDervDepth,
                             const Float sigma,
                             const Float lensSigma)
    : h2mcParam(sigma), lensH2mcParam(lensSigma) {
    sceneParams.resize(GetSceneSerializedSize());
    Serialize(scene, &sceneParams[0]);
    ssubPath.primary.resize(GetPrimaryParamSize(maxDervDepth, maxDervDepth));
    ssubPath.vertParams.resize(GetVertParamSize(maxDervDepth, maxDervDepth));
}

Float H2MCSmallStep::Mutate(const MLTState &mltState,
                            const Float normalization,
                            MarkovState &currentState,
                            MarkovState &proposalState,
                            RNG &rng) {
    const Scene *scene = mltState.scene;
    // Sometimes the derivatives are noisy so that the light paths
    // will "stuck" in some regions, we probabilistically switch to
    // uniform sampling to avoid stucking
    std::uniform_real_distribution<Float> uniDist(Float(0.0), Float(1.0));
    if (uniDist(rng) < scene->options->uniformMixingProbability) {
        Float a =
            isotropicSmallStep.Mutate(mltState, normalization, currentState, proposalState, rng);
        lastMutationType = isotropicSmallStep.lastMutationType;
        return a;
    }

    spContribs.clear();

    const Float lensPerturbProb = scene->options->lensPerturbProb;
    Float a = Float(1.0);
    assert(currentState.valid);
    if (currentState.spContrib.lensScore > Float(0.0) && uniDist(rng) < lensPerturbProb) {
        lastMutationType = MutationType::H2MCLens;
        auto initLensGaussian = [&](MarkovState &state) {
            const SubpathContrib &cspContrib = state.spContrib;
            auto funcIt =
                mltState.lensFuncDervMap.find({cspContrib.camDepth, cspContrib.lightDepth});
            if (funcIt != mltState.lensFuncDervMap.end()) {
                vGrad.resize(2, Float(0.0));
                vHess.resize(2 * 2, Float(0.0));
                if (cspContrib.lensScore > Float(1e-15)) {
                    PathFuncDerv dervFunc = funcIt->second;
                    Serialize(scene, state.path, ssubPath);
                    dervFunc(&cspContrib.screenPos[0],
                             &ssubPath.primary[0],
                             &sceneParams[0],
                             &ssubPath.vertParams[0],
                             &vGrad[0],
                             &vHess[0]);
                    if (!IsFinite(vGrad) || !IsFinite(vHess)) {
                        // Usually caused by floating point round-off error
                        // (or, of course, bugs)
                        std::fill(vGrad.begin(), vGrad.end(), Float(0.0));
                        std::fill(vHess.begin(), vHess.end(), Float(0.0));
                    }

                    assert(IsFinite(vGrad));
                    assert(IsFinite(vHess));
                }
                ComputeGaussian(
                    lensH2mcParam, cspContrib.ssScore, vGrad, vHess, state.lensGaussian);
            } else {
                IsotropicGaussian(2, lensH2mcParam.sigma, state.lensGaussian);
            }
            state.lensGaussianInitialized = true;
        };

        if (!currentState.lensGaussianInitialized) {
            initLensGaussian(currentState);
        }

        assert(currentState.lensGaussianInitialized);

        Vector offset(2);
        GenerateSample(currentState.lensGaussian, offset, rng);
        Vector screenPos(2);
        screenPos[0] = Modulo(currentState.spContrib.screenPos[0] + offset[0], Float(1.0));
        screenPos[1] = Modulo(currentState.spContrib.screenPos[1] + offset[1], Float(1.0));
        proposalState.path = currentState.path;
        const auto perturbLensFunc = mltState.perturbLensFunc;
        perturbLensFunc(scene, screenPos, proposalState.path, spContribs);
        SetIsMoving(proposalState.path);
        proposalState.gaussianInitialized = false;
        if (spContribs.size() > 0) {
            assert(spContribs.size() == 1);
            proposalState.spContrib = spContribs[0];
            initLensGaussian(proposalState);
            const Float py = GaussianLogPdf(offset, currentState.lensGaussian);
            const Float px = GaussianLogPdf(-offset, proposalState.lensGaussian);
            a = Clamp(expf(px - py) * proposalState.spContrib.lensScore /
                          currentState.spContrib.lensScore,
                      Float(0.0),
                      Float(1.0));
            proposalState.toSplat.clear();
            for (const auto &spContrib : spContribs) {
                proposalState.toSplat.push_back(SplatSample{
                    spContrib.screenPos, spContrib.contrib * (normalization / spContrib.lsScore)});
            }
        } else {
            a = Float(0.0);
        }
    } else {
        lastMutationType = MutationType::H2MCSmall;
        const auto perturbPathFunc = mltState.perturbPathFunc;
        const int dim = GetDimension(currentState.path);
        auto initGaussian = [&](MarkovState &state) {
            const SubpathContrib &cspContrib = state.spContrib;
            const auto &fmap =
                state.path.isMoving ? mltState.funcDervMap : mltState.staticFuncDervMap;
            auto funcIt = fmap.find({cspContrib.camDepth, cspContrib.lightDepth});
            const int dim = GetDimension(state.path);
            if (funcIt != fmap.end()) {
                vGrad.resize(dim, Float(0.0));
                vHess.resize(dim * dim, Float(0.0));
                if (cspContrib.ssScore > Float(1e-15)) {
                    PathFuncDerv dervFunc = funcIt->second;
                    assert(dervFunc != nullptr);
                    Serialize(scene, state.path, ssubPath);
                    dervFunc(&cspContrib.screenPos[0],
                             &ssubPath.primary[0],
                             &sceneParams[0],
                             &ssubPath.vertParams[0],
                             &vGrad[0],
                             &vHess[0]);
                    if (!IsFinite(vGrad) || !IsFinite(vHess)) {
                        // Usually caused by floating point round-off error
                        // (or, of course, bugs)
                        std::fill(vGrad.begin(), vGrad.end(), Float(0.0));
                        std::fill(vHess.begin(), vHess.end(), Float(0.0));
                    }
                    assert(IsFinite(vGrad));
                    assert(IsFinite(vHess));
                }
                ComputeGaussian(h2mcParam, cspContrib.ssScore, vGrad, vHess, state.gaussian);
            } else {
                IsotropicGaussian(dim, h2mcParam.sigma, state.gaussian);
            }
            state.gaussianInitialized = true;
        };

        if (!currentState.gaussianInitialized) {
            initGaussian(currentState);
        }

        assert(currentState.gaussianInitialized);

        Vector offset(dim);
        GenerateSample(currentState.gaussian, offset, rng);
        proposalState.path = currentState.path;
        perturbPathFunc(scene, offset, proposalState.path, spContribs, rng);
        SetIsMoving(proposalState.path);
        proposalState.lensGaussianInitialized = false;
        if (spContribs.size() > 0) {
            assert(spContribs.size() == 1);
            proposalState.spContrib = spContribs[0];
            initGaussian(proposalState);
            Float py = GaussianLogPdf(offset, currentState.gaussian);
            Float px = Float(0.0);
            if (currentState.path.isMoving == proposalState.path.isMoving) {
                px = GaussianLogPdf(-offset, proposalState.gaussian);
            } else {
                Float timeOffset = proposalState.path.time - currentState.path.time;
                if (timeOffset > Float(0.5)) {
                    timeOffset -= Float(1.0);
                } else if (timeOffset < Float(-0.5)) {
                    timeOffset += Float(1.0);
                }
                const Float timeStdDev = scene->options->discreteStdDev;
                const Float timeGaussianLogPdf =
                    -square(timeOffset) / (Float(2.0) * square(timeStdDev)) -
                    (timeStdDev * sqrt(Float(2.0) * c_PI));
                if (proposalState.path.isMoving) {
                    Vector extendedOffset(dim + 1);
                    extendedOffset[0] = timeOffset;
                    extendedOffset.tail(dim) = offset;
                    px = GaussianLogPdf(-extendedOffset, proposalState.gaussian);
                    assert(!currentState.path.isMoving);
                    py += timeGaussianLogPdf;
                } else {
                    assert(currentState.path.isMoving);
                    px = GaussianLogPdf(offset.tail(dim - 1), proposalState.gaussian) +
                         timeGaussianLogPdf;
                }
            }
            a = Clamp(std::exp(px - py) * proposalState.spContrib.ssScore /
                          currentState.spContrib.ssScore,
                      Float(0.0),
                      Float(1.0));
            proposalState.toSplat.clear();
            for (const auto &spContrib : spContribs) {
                proposalState.toSplat.push_back(SplatSample{
                    spContrib.screenPos, spContrib.contrib * (normalization / spContrib.lsScore)});
            }
        } else {
            a = Float(0.0);
        }
    }

    return a;
}

void MLT(const Scene *scene, const std::shared_ptr<const PathFuncLib> pathFuncLib) {
    const MLTState mltState{scene,
                            scene->options->bidirectional ? GeneratePathBidir : GeneratePath,
                            scene->options->bidirectional ? PerturbPathBidir : PerturbPath,
                            scene->options->bidirectional ? PerturbLensBidir : PerturbLens,
                            pathFuncLib->funcMap,
                            pathFuncLib->dervFuncMap,
                            pathFuncLib->staticFuncMap,
                            pathFuncLib->staticDervFuncMap,
                            pathFuncLib->lensFuncMap,
                            pathFuncLib->lensDervFuncMap};
    const int spp = scene->options->spp;
    std::shared_ptr<const Camera> camera = scene->camera;
    const Float largeStepProb = scene->options->largeStepProb;
    std::shared_ptr<Image3> film = camera->film;
    film->Clear();
    const int pixelHeight = GetPixelHeight(camera.get());
    const int pixelWidth = GetPixelWidth(camera.get());
    SampleBuffer directBuffer(pixelWidth, pixelHeight);
    DirectLighting(scene, directBuffer);

    const int64_t numPixels = int64_t(pixelWidth) * int64_t(pixelHeight);
    const int64_t totalSamples = int64_t(spp) * numPixels;
    const int64_t numChains = scene->options->numChains;
    const int64_t numSamplesPerChain = totalSamples / numChains;
    const int64_t chainsNeedExtraSamples = numSamplesPerChain % numChains;

    std::vector<MarkovState> initStates;
    std::shared_ptr<PiecewiseConstant1D> lengthDist;
    const Float avgScore =
        MLTInit(mltState, scene->options->numInitSamples, numChains, initStates, lengthDist);
    std::cout << "Average brightness:" << avgScore << std::endl;
    const Float normalization = avgScore;

    ProgressReporter reporter(totalSamples);
    const int reportInterval = 16384;
    int intervalImgId = 1;

    std::atomic<int64_t> largeStepAccepted(0), largeStepTotal(0);
    std::atomic<int64_t> smallStepAccepted(0), smallStepTotal(0);
    std::atomic<int64_t> lensStepAccepted(0), lensStepTotal(0);
    std::atomic<int64_t> h2mcSmallStepAccepted(0), h2mcSmallStepTotal(0);
    std::atomic<int64_t> h2mcLensStepAccepted(0), h2mcLensStepTotal(0);

    SampleBuffer indirectBuffer(pixelWidth, pixelHeight);
    Timer timer;
    Tick(timer);
    ParallelFor([&](const int chainId) {
        const int seed = chainId + scene->options->seedOffset;
        RNG rng(seed);
        std::uniform_real_distribution<Float> uniDist(Float(0.0), Float(1.0));

        int64_t numSamplesThisChain =
            numSamplesPerChain + ((chainId < chainsNeedExtraSamples) ? 1 : 0);
        std::vector<Float> contribCdf;
        MarkovState currentState = initStates[chainId];
        MarkovState proposalState{false};

        std::unique_ptr<LargeStep> largeStep =
            std::unique_ptr<LargeStep>(new LargeStep(lengthDist));
        std::unique_ptr<Mutation> smallStep =
            scene->options->h2mc
                ? std::unique_ptr<Mutation>(new H2MCSmallStep(scene,
                                                              pathFuncLib->maxDepth,
                                                              scene->options->perturbStdDev,
                                                              scene->options->lensPerturbStdDev))
                : std::unique_ptr<Mutation>(new SmallStep());

        for (int sampleIdx = 0; sampleIdx < numSamplesThisChain; sampleIdx++) {
            Float a = Float(1.0);
            bool isLargeStep = false;
            if (!currentState.valid || uniDist(rng) < largeStepProb) {
                isLargeStep = true;
                a = largeStep->Mutate(mltState, normalization, currentState, proposalState, rng);
            } else {
                a = smallStep->Mutate(mltState, normalization, currentState, proposalState, rng);
            }


            if (currentState.valid && a < Float(1.0)) {
                for (const auto splat : currentState.toSplat) {
                    Splat(indirectBuffer, splat.screenPos, (Float(1.0) - a) * splat.contrib);
                }
            }
            if (a > Float(0.0)) {
                for (const auto splat : proposalState.toSplat) {
                    Splat(indirectBuffer, splat.screenPos, a * splat.contrib);
                }
            }

            if (a > Float(0.0) && uniDist(rng) <= a) {
                ToSubpath(proposalState.spContrib.camDepth,
                          proposalState.spContrib.lightDepth,
                          proposalState.path);
                std::swap(currentState, proposalState);
                currentState.valid = true;
                if (isLargeStep) {
                    largeStep->lastScoreSum = currentState.scoreSum;
                    largeStep->lastScore = currentState.spContrib.lsScore;
                    currentState.lensGaussianInitialized = false;
                    currentState.gaussianInitialized = false;
                    largeStepAccepted++;
                } else {
                    if (smallStep->lastMutationType == MutationType::Small) {
                        smallStepAccepted++;
                    } else if (smallStep->lastMutationType == MutationType::H2MCSmall) {
                        h2mcSmallStepAccepted++;
                    } else if (smallStep->lastMutationType == MutationType::Lens) {
                        lensStepAccepted++;
                    } else {
                        assert(smallStep->lastMutationType == MutationType::H2MCLens);
                        h2mcLensStepAccepted++;
                    }
                }
            }

            if (isLargeStep) {
                largeStepTotal++;
            } else {
                if (smallStep->lastMutationType == MutationType::Small) {
                    smallStepTotal++;
                } else if (smallStep->lastMutationType == MutationType::H2MCSmall) {
                    h2mcSmallStepTotal++;
                } else if (smallStep->lastMutationType == MutationType::Lens) {
                    lensStepTotal++;
                } else {
                    assert(smallStep->lastMutationType == MutationType::H2MCLens);
                    h2mcLensStepTotal++;
                }
            }

            if (sampleIdx > 0 && (sampleIdx % reportInterval == 0)) {
                reporter.Update(reportInterval);
                const int reportIntervalSpp = scene->options->reportIntervalSpp;
                if (threadIndex == 0 && reportIntervalSpp > 0) {
                    if (reporter.GetWorkDone() >
                        uint64_t(numPixels * reportIntervalSpp * intervalImgId)) {
                        BufferToFilm(indirectBuffer,
                                     film.get(),
                                     inverse(Float(reportIntervalSpp * intervalImgId)));
                        WriteImage("intermediate_" + std::to_string(intervalImgId) + ".exr",
                                   film.get());
                        intervalImgId++;
                    }
                }
            }
        }
        reporter.Update(numSamplesThisChain % reportInterval);
    }, numChains);
    TerminateWorkerThreads();
    reporter.Done();
    Float elapsed = Tick(timer);
    std::cout << "Elapsed time:" << elapsed << std::endl;

    std::cout << "Large step acceptance rate:" << Float(largeStepAccepted) / Float(largeStepTotal)
              << "(" << int64_t(largeStepAccepted) << "/" << int64_t(largeStepTotal) << ")"
              << std::endl;

    std::cout << "Small step acceptance rate:" << Float(smallStepAccepted) / Float(smallStepTotal)
              << "(" << int64_t(smallStepAccepted) << "/" << int64_t(smallStepTotal) << ")"
              << std::endl;

    std::cout << "Lens step acceptance rate:" << Float(lensStepAccepted) / Float(lensStepTotal)
              << "(" << int64_t(lensStepAccepted) << "/" << int64_t(lensStepTotal) << ")"
              << std::endl;

    std::cout << "H2MC Small step acceptance rate:"
              << Float(h2mcSmallStepAccepted) / Float(h2mcSmallStepTotal) << "("
              << int64_t(h2mcSmallStepAccepted) << "/" << int64_t(h2mcSmallStepTotal) << ")"
              << std::endl;

    std::cout << "H2MC Lens step acceptance rate:"
              << Float(h2mcLensStepAccepted) / Float(h2mcLensStepTotal) << "("
              << int64_t(h2mcLensStepAccepted) << "/" << int64_t(h2mcLensStepTotal) << ")"
              << std::endl;


    SampleBuffer buffer(pixelWidth, pixelHeight);
    Float directWeight =
        scene->options->directSpp > 0 ? inverse(Float(scene->options->directSpp)) : Float(0.0);
    Float indirectWeight = spp > 0 ? inverse(Float(spp)) : Float(0.0);
    MergeBuffer(directBuffer, directWeight, indirectBuffer, indirectWeight, buffer);
    BufferToFilm(buffer, film.get());
}
