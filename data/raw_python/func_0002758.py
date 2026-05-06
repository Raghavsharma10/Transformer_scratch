def set_nvidia_environment_variables(environment, gpu_ids):
    """
    Updates a dictionary containing environment variables to setup Nvidia-GPUs.

    :param environment: The environment variables to update
    :param gpu_ids: A list of GPU ids
    """

    if gpu_ids:
        nvidia_visible_devices = ""
        for gpu_id in gpu_ids:
            nvidia_visible_devices += "{},".format(gpu_id)
        environment["NVIDIA_VISIBLE_DEVICES"] = nvidia_visible_devices