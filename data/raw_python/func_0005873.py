def _compiler_type():
    """
    Gets the compiler type from distutils. On Windows with MSVC it will be
    "msvc". On macOS and linux it is "unix".

    Borrowed from https://github.com/pyca/cryptography/blob\
        /05b34433fccdc2fec0bb014c3668068169d769fd/src/_cffi_src/utils.py#L78
    """
    dist = distutils.dist.Distribution()
    dist.parse_config_files()
    cmd = dist.get_command_obj('build')
    cmd.ensure_finalized()
    compiler = distutils.ccompiler.new_compiler(compiler=cmd.compiler)
    return compiler.compiler_type