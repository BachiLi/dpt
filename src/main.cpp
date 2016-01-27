#include "parsescene.h"
#include "pathtrace.h"
#include "mlt.h"
#include "image.h"
#include "camera.h"
#include "texturesystem.h"
#include "parallel.h"
#include "path.h"

#include <iostream>
#include <string>

void DptInit() {
    TextureSystem::Init();
    std::cout << "Running with " << NumSystemCores() << " threads." << std::endl;
    if (getenv("DPT_LIBPATH") == nullptr) {
        std::cout
            << "Environment variable DPT_LIBPATH not set."
               "Please set it to a directory (e.g. export DPT_LIBPATH=/dir/to/dpt/src/bin) so "
               "that I can write/find the dynamic libraries." << std::endl;
        exit(0);
    }
}

void DptCleanup() {
    TextureSystem::Destroy();
    TerminateWorkerThreads();
}

int main(int argc, char *argv[]) {
    if (argc <= 1) {
        return 0;
    }

    try {
        DptInit();

        bool compilePathLib = false;
        bool compileBidirPathLib = false;
        int maxDervDepth = 6;
        std::vector<std::string> filenames;
        for (int i = 1; i < argc; ++i) {
            if (std::string(argv[i]) == "--compile-pathlib") {
                compilePathLib = true;
            } else if (std::string(argv[i]) == "--compile-bidirpathlib") {
                compileBidirPathLib = true;
            } else if (std::string(argv[i]) == "--max-derivatives-depth") {
                maxDervDepth = std::stoi(std::string(argv[++i]));
            } else {
                filenames.push_back(std::string(argv[i]));
            }
        }

        if (compilePathLib) {
            CompilePathFuncLibrary(false, maxDervDepth);
        }
        if (compileBidirPathLib) {
            CompilePathFuncLibrary(true, maxDervDepth);
        }

        std::string cwd = getcwd(NULL, 0);
        for (std::string filename : filenames) {
            if (filename.rfind('/') != std::string::npos &&
                chdir(filename.substr(0, filename.rfind('/')).c_str()) != 0) {
                Error("chdir failed");
            }
            if (filename.rfind('/') != std::string::npos) {
                filename = filename.substr(filename.rfind('/') + 1);
            }
            std::unique_ptr<const Scene> scene = ParseScene(filename);
            std::string integrator = scene->options->integrator;
            if (integrator == "mc") {
                std::shared_ptr<const PathFuncLib> library =
                    BuildPathFuncLibrary(scene->options->bidirectional, maxDervDepth);
                PathTrace(scene.get(), library);
            } else if (integrator == "mcmc") {
                std::shared_ptr<const PathFuncLib> library =
                    BuildPathFuncLibrary(scene->options->bidirectional, maxDervDepth);
                MLT(scene.get(), library);
            } else {
                Error("Unknown integrator");
            }
            WriteImage(scene->outputName, GetFilm(scene->camera.get()).get());
            if (chdir(cwd.c_str()) != 0) {
                Error("chdir failed");
            }
        }
        DptCleanup();
    } catch (std::exception &ex) {
        std::cerr << ex.what() << std::endl;
    }

    return 0;
}
