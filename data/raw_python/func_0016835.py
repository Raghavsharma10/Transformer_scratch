def nvcc_compiler_settings():
    """ Find nvcc and the CUDA installation """

    search_paths = os.environ.get('PATH', '').split(os.pathsep)
    nvcc_path = find_in_path('nvcc', search_paths)
    default_cuda_path = os.path.join('usr', 'local', 'cuda')
    cuda_path = os.environ.get('CUDA_PATH', default_cuda_path)

    nvcc_found = os.path.exists(nvcc_path)
    cuda_path_found = os.path.exists(cuda_path)

    # Can't find either NVCC or some CUDA_PATH
    if not nvcc_found and not cuda_path_found:
        raise InspectCudaException("Neither nvcc '{}' "
            "or the CUDA_PATH '{}' were found!".format(
                nvcc_path, cuda_path))

    # No NVCC, try find it in the CUDA_PATH
    if not nvcc_found:
        log.warn("nvcc compiler not found at '{}'. "
            "Searching within the CUDA_PATH '{}'"
                .format(nvcc_path, cuda_path))

        bin_dir = os.path.join(cuda_path, 'bin')
        nvcc_path = find_in_path('nvcc', bin_dir)
        nvcc_found = os.path.exists(nvcc_path)

        if not nvcc_found:
            raise InspectCudaException("nvcc not found in '{}' "
                "or under the CUDA_PATH at '{}' "
                .format(search_paths, cuda_path))

    # No CUDA_PATH found, infer it from NVCC
    if not cuda_path_found:
        cuda_path = os.path.normpath(
            os.path.join(os.path.dirname(nvcc_path), ".."))

        log.warn("CUDA_PATH not found, inferring it as '{}' "
            "from the nvcc location '{}'".format(
                cuda_path, nvcc_path))

        cuda_path_found = True

    # Set up the compiler settings
    include_dirs = []
    library_dirs = []
    define_macros = []

    if cuda_path_found:
        include_dirs.append(os.path.join(cuda_path, 'include'))
        if sys.platform == 'win32':
            library_dirs.append(os.path.join(cuda_path, 'bin'))
            library_dirs.append(os.path.join(cuda_path, 'lib', 'x64'))
        else:
            library_dirs.append(os.path.join(cuda_path, 'lib64'))
            library_dirs.append(os.path.join(cuda_path, 'lib'))
    if sys.platform == 'darwin':
        library_dirs.append(os.path.join(default_cuda_path, 'lib'))

    return {
        'cuda_available' : True,
        'nvcc_path' : nvcc_path,
        'include_dirs': include_dirs,
        'library_dirs': library_dirs,
        'define_macros': define_macros,
        'libraries' : ['cudart', 'cuda'],
        'language': 'c++',
    }