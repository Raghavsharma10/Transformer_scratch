def create_tensorflow_extension(nvcc_settings, device_info):
    """ Create an extension that builds the custom tensorflow ops """
    import tensorflow as tf
    import glob

    use_cuda = (bool(nvcc_settings['cuda_available'])
        and tf.test.is_built_with_cuda())

    # Source and includes
    source_path = os.path.join('montblanc', 'impl', 'rime', 'tensorflow', 'rime_ops')
    sources = glob.glob(os.path.join(source_path, '*.cpp'))

    # Header dependencies
    depends = glob.glob(os.path.join(source_path, '*.h'))

    # Include directories
    tf_inc = tf.sysconfig.get_include()
    include_dirs = [os.path.join('montblanc', 'include'), source_path]
    include_dirs += [tf_inc, os.path.join(tf_inc, "external", "nsync", "public")]

    # Libraries
    library_dirs = [tf.sysconfig.get_lib()]
    libraries = ['tensorflow_framework']
    extra_link_args = ['-fPIC', '-fopenmp', '-g0']

    # Macros
    define_macros = [
        ('_MWAITXINTRIN_H_INCLUDED', None),
        ('_FORCE_INLINES', None),
        ('_GLIBCXX_USE_CXX11_ABI', 0)]

    # Common flags
    flags = ['-std=c++11']

    gcc_flags = flags + ['-g0', '-fPIC', '-fopenmp', '-O2']
    gcc_flags += ['-march=native', '-mtune=native']
    nvcc_flags = flags + []

    # Add cuda specific build information, if it is available
    if use_cuda:
        # CUDA source files
        sources += glob.glob(os.path.join(source_path, '*.cu'))
        # CUDA include directories
        include_dirs += nvcc_settings['include_dirs']
        # CUDA header dependencies
        depends += glob.glob(os.path.join(source_path, '*.cuh'))
        # CUDA libraries
        library_dirs += nvcc_settings['library_dirs']
        libraries += nvcc_settings['libraries']
        # Flags
        nvcc_flags += ['-x', 'cu']
        nvcc_flags += ['--compiler-options', '"-fPIC"']
        # --gpu-architecture=sm_xy flags
        nvcc_flags += cuda_architecture_flags(device_info)
        # Ideally this would be set in define_macros, but
        # this must be set differently for gcc and nvcc
        nvcc_flags += ['-DGOOGLE_CUDA=%d' % int(use_cuda)]

    return Extension(tensorflow_extension_name,
        sources=sources,
        include_dirs=include_dirs,
        depends=depends,
        library_dirs=library_dirs,
        libraries=libraries,
        define_macros=define_macros,
        # this syntax is specific to this build system
        # we're only going to use certain compiler args with nvcc and not with gcc
        # the implementation of this trick is in customize_compiler_for_nvcc() above
        extra_compile_args={ 'gcc': gcc_flags, 'nvcc': nvcc_flags },
        extra_link_args=extra_link_args,
    )