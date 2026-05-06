def extensions():
    """Returns list of `cython` extensions for `lazy_cythonize`."""
    import numpy
    from Cython.Build import cythonize
    ext = [
            Extension('phydmslib.numutils', ['phydmslib/numutils.pyx'],
                    include_dirs=[numpy.get_include()],
                    extra_compile_args=['-Wno-unused-function']),
          ]      
    return cythonize(ext)