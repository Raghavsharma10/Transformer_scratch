def get_cuda_devices():
    """
    Imports pycuda at runtime and reads GPU information.
    :return: A list of available cuda GPUs.
    """

    devices = []

    try:
        import pycuda.autoinit
        import pycuda.driver as cuda

        for device_id in range(cuda.Device.count()):
            vram = cuda.Device(device_id).total_memory()

            devices.append(GPUDevice(device_id, vram))
    except ImportError:
        raise InsufficientGPUError('No Nvidia-GPUs could be found, because "pycuda" could not be imported.')

    return devices