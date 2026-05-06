def inspect_cuda():
    """ Return cuda device information and nvcc/cuda setup """
    nvcc_settings = nvcc_compiler_settings()
    sysconfig.get_config_vars()
    nvcc_compiler = ccompiler.new_compiler()
    sysconfig.customize_compiler(nvcc_compiler)
    customize_compiler_for_nvcc(nvcc_compiler, nvcc_settings)

    output = inspect_cuda_version_and_devices(nvcc_compiler, nvcc_settings)

    return json.loads(output), nvcc_settings