def get_virtualenv_env_data(mgr):
    """Finds kernel specs from virtualenv environments

    env_data is a structure {name -> (resourcedir, kernel spec)}
    """

    if not mgr.find_virtualenv_envs:
        return {}

    mgr.log.debug("Looking for virtualenv environments in %s...", mgr.virtualenv_env_dirs)

    # find all potential env paths
    env_paths = find_env_paths_in_basedirs(mgr.virtualenv_env_dirs)

    mgr.log.debug("Scanning virtualenv environments for python kernels...")
    env_data = convert_to_env_data(mgr=mgr,
                                   env_paths=env_paths,
                                   validator_func=validate_IPykernel,
                                   activate_func=_get_env_vars_for_virtualenv_env,
                                   name_template=mgr.virtualenv_prefix_template,
                                   display_name_template=mgr.display_name_template,
                                   # virtualenv has only python, so no need for a prefix
                                   name_prefix="")
    return env_data