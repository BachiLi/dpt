#include "gaussian.h"

void IsotropicGaussian(const int dim, const Float sigma, Gaussian &gaussian) {
    auto vsigma = Vector::Constant(dim, sigma);
    auto invSigmaSq = vsigma.cwiseProduct(vsigma).cwiseInverse();
    gaussian.mean = Vector::Zero(dim);
    gaussian.covL = vsigma.asDiagonal();
    gaussian.invCov = invSigmaSq.asDiagonal();
    gaussian.logDet = Float(0.0);
    for (int i = 0; i < dim; i++) {
        gaussian.logDet += log(invSigmaSq[i]);
    }
}

Float GaussianLogPdf(const Vector &offset, const Gaussian &gaussian) {
    assert(gaussian.mean.size() == offset.size());
    auto d = offset - gaussian.mean;
    Float logPdf = gaussian.mean.size() * (-Float(0.5) * log(Float(2.0 * M_PI)));
    logPdf += Float(0.5) * gaussian.logDet;
    logPdf -= Float(0.5) * (d.transpose() * (gaussian.invCov * d))[0];
    return logPdf;
}

void GenerateSample(const Gaussian &gaussian, Vector &x, RNG &rng) {
    if (gaussian.mean.size() != x.size()) {
        std::cerr << "gaussian.mean.size():" << gaussian.mean.size() << std::endl;
        std::cerr << "x.size():" << x.size() << std::endl;
    }
    assert(gaussian.mean.size() == x.size());
    std::normal_distribution<Float> normDist(Float(0.0), Float(1.0));
    for (int i = 0; i < x.size(); i++) {
        x[i] = normDist(rng);
    }
    x = gaussian.covL * x + gaussian.mean;
}
