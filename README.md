# dpt

dpt is a prototypical renderer that implements the algorithm presented in the paper "[Anisotropic Gaussian Mutations for Metropolis Light Transport through Hessian-Hamiltonian Dynamics] (https://people.csail.mit.edu/tzumao/h2mc/)" by Tzu-Mao Li, Jaakko Lehtinen, Ravi Ramamoorthi, Wenzel Jakob, and Frédo Durand. The algorithm utilizes the derivatives of the contribution function of a (bidirectional) path tracer to guide local sampling, hence the name dpt.

dpt supports a limited form of [mitsuba](http://www.mitsuba-renderer.org/)'s scene format.  It supports pinhole camera, three kinds of BSDF (diffuse, phong, roughdielectric), three kinds of emitters (point, area, envmap), trianglemesh shape, and linear motion blur.  See scenes/torus for an example scene.

If you want to understand the algorithm by looking at the source code, a good starting point is to look at mlt.cpp, h2mc.cpp, and path.cpp first.

dpt uses [tup](http://gittup.org/tup/index.html) as its build system.  It depends on several libraries: [Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page), [OpenImageIO](https://github.com/OpenImageIO/oiio), [embree](https://embree.github.io/), and [zlib](http://www.zlib.net/).  It uses [ispc](https://ispc.github.io/ispc.html) to compile the derivative code it generates.  It also uses [pugixml](http://pugixml.org/) to parse the scene file and [PCG](http://www.pcg-random.org/) for fast and high-quality random number generation.  A large portion of dpt is inspired by [mitsuba](http://www.mitsuba-renderer.org/), [pbrt-v3](https://github.com/mmp/pbrt-v3), and [SmallVCM](https://github.com/SmallVCM/SmallVCM/)

dpt needs to generate a dynamic library that contains functions for computing the derivatives of the path contribution function.  Before you execute dpt, you will need to specify the directory used for reading/writing the dynamic library by setting the environment variable DPT\_LIBPATH (e.g. export DPT\_LIBPATH=/path/to/dpt/src/bin).  dpt will search that directory for the dynamic library, and if it does not find it, it will create one.

The program is only tested on OSX 10.10.5 with clang and Ubuntu 14.04 with gcc.  Currently it does not support windows system.

Please contact Tzu-Mao Li (tzumao at mit.edu) if there are any issues/comments/questions.

