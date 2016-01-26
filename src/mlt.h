#pragma once

#include "scene.h"

#include <memory>

struct PathFuncLib;

void MLT(const Scene *scene, const std::shared_ptr<const PathFuncLib> pathFuncLib);
