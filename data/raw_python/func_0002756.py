def match_gpus(available_devices, requirements):
    """
    Determines sufficient GPUs for the given requirements and returns a list of GPUDevices.
    If there aren't sufficient GPUs a InsufficientGPUException is thrown.

    :param available_devices: A list of GPUDevices
    :param requirements: A list of GPURequirements

    :return: A list of sufficient devices
    """

    if not requirements:
        return []

    if not available_devices:
        raise InsufficientGPUError("No GPU devices available, but {} devices required.".format(len(requirements)))

    available_devices = available_devices.copy()

    used_devices = []

    for req in requirements:
        dev = search_device(req, available_devices)
        if dev:
            used_devices.append(dev)
            available_devices.remove(dev)
        else:
            raise InsufficientGPUError("Not all GPU requirements could be fulfilled.")

    return used_devices