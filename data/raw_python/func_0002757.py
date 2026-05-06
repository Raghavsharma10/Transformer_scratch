def get_gpu_requirements(gpus_reqs):
    """
    Extracts the GPU from a dictionary requirements as list of GPURequirements.

    :param gpus_reqs: A dictionary {'count': <count>} or a list [{min_vram: <min_vram>}, {min_vram: <min_vram>}, ...]
    :return: A list of GPURequirements
    """
    requirements = []

    if gpus_reqs:
        if type(gpus_reqs) is dict:
            count = gpus_reqs.get('count')
            if count:
                for i in range(count):
                    requirements.append(GPURequirement())
        elif type(gpus_reqs) is list:
            for gpu_req in gpus_reqs:
                requirements.append(GPURequirement(min_vram=gpu_req['minVram']))
        return requirements
    else:
        # If no requirements are supplied
        return []