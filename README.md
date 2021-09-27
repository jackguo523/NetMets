NetMets: software for quantifying and visualizing errors in biological network segmentation (https://bmcbioinformatics.biomedcentral.com/articles/10.1186/1471-2105-13-S8-S7)

NetMets can bebuilt using CMake (https://cmake.org/) and a C/C++ compiler.

The STIM codebase is required, but will be cloned automatically if Git (https://git-scm.com/) is installed. The codebase can be downloaded manually here: https://git.stim.ee.uh.edu/codebase/stimlib

Required libraries: OpenGL: http://www.opengl.org/ and GLEW: http://glew.sourceforge.net/


Step-by-step instructions:

1) Download and install CMake
2) Download and install OpenGL
3) Download and install GLEW
4) Download and install Git
5) Set the CMake source directory to the directory containing this file
6) Specify the CMake build directory where you want the executable built
7) Use CMake to Configure and Generate the build environment
8) Build the software (ex. in Visual Studio you will open the generated solution and compile)