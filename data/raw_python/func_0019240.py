def compile_(self):
        """Translate cython code to C code and compile it."""
        from Cython import Build
        argv = copy.deepcopy(sys.argv)
        sys.argv = [sys.argv[0], 'build_ext', '--build-lib='+self.buildpath]
        exc_modules = [
                distutils.extension.Extension(
                        'hydpy.cythons.autogen.'+self.cyname,
                        [self.pyxfilepath], extra_compile_args=['-O2'])]
        distutils.core.setup(ext_modules=Build.cythonize(exc_modules),
                             include_dirs=[numpy.get_include()])
        sys.argv = argv