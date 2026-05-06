def customize_compiler_for_nvcc(compiler, nvcc_settings):
    """inject deep into distutils to customize gcc/nvcc dispatch """

    # tell the compiler it can process .cu files
    compiler.src_extensions.append('.cu')

    # save references to the default compiler_so and _compile methods
    default_compiler_so = compiler.compiler_so
    default_compile = compiler._compile

    # now redefine the _compile method. This gets executed for each
    # object but distutils doesn't have the ability to change compilers
    # based on source extension: we add it.
    def _compile(obj, src, ext, cc_args, extra_postargs, pp_opts):
        # Use NVCC for .cu files
        if os.path.splitext(src)[1] == '.cu':
            compiler.set_executable('compiler_so', nvcc_settings['nvcc_path'])

        default_compile(obj, src, ext, cc_args, extra_postargs, pp_opts)
        # reset the default compiler_so, which we might have changed for cuda
        compiler.compiler_so = default_compiler_so

    # inject our redefined _compile method into the class
    compiler._compile = _compile