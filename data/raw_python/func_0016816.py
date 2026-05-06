def cuda_architecture_flags(device_info):
    """
    Emit a list of architecture flags for each CUDA device found
    ['--gpu-architecture=sm_30', '--gpu-architecture=sm_52']
    """
    # Figure out the necessary device architectures
    if len(device_info['devices']) == 0:
        archs = ['--gpu-architecture=sm_30']
        log.info("No CUDA devices found, defaulting to architecture '{}'".format(archs[0]))
    else:
        archs = set()

        for device in device_info['devices']:
            arch_str = '--gpu-architecture=sm_{}{}'.format(device['major'], device['minor'])
            log.info("Using '{}' for '{}'".format(arch_str, device['name']))
            archs.add(arch_str)

    return list(archs)