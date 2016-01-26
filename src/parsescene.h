#pragma once

#include "scene.h"

#include <string>
#include <memory>

std::unique_ptr<const Scene> ParseScene(const std::string &filename);
