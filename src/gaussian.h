#pragma once

#include "commondef.h"
#include <algorithm>

struct Gaussian {
    Matrix covL;
    Matrix invCov;
    Vector mean;
    Float logDet;
};

inline int GetDimension(const Gaussian &gaussian) {
    return gaussian.mean.size();
}

void IsotropicGaussian(const int dim, const Float sigma, Gaussian &gaussian);
Float GaussianLogPdf(const Vector &offset, const Gaussian &gaussian);
void GenerateSample(const Gaussian &gaussian, Vector &x, RNG &rng);
