def get_conda_env_data(mgr):
    """Finds kernel specs from conda environments

    env_data is a structure {name -> (resourcedir, kernel spec)}
    """
    if not mgr.find_conda_envs:
        return {}

    mgr.log.debug("Looking for conda environments in %s...", mgr.conda_env_dirs)

    # find all potential env paths
    env_paths = find_env_paths_in_basedirs(mgr.conda_env_dirs)
    env_paths.extend(_find_conda_env_paths_from_conda(mgr))
    env_paths = list(set(env_paths)) # remove duplicates

    mgr.log.debug("Scanning conda environments for python kernels...")
    env_data = convert_to_env_data(mgr=mgr,
                                   env_paths=env_paths,
                                   validator_func=validate_IPykernel,
                                   activate_func=_get_env_vars_for_conda_env,
                                   name_template=mgr.conda_prefix_template,
                                   display_name_template=mgr.display_name_template,
                                   name_prefix="")  # lets keep the py kernels without a prefix...
    if mgr.find_r_envs:
        mgr.log.debug("Scanning conda environments for R kernels...")
        env_data.update(convert_to_env_data(mgr=mgr,
                                            env_paths=env_paths,
                                            validator_func=validate_IRkernel,
                                            activate_func=_get_env_vars_for_conda_env,
                                            name_template=mgr.conda_prefix_template,
                                            display_name_template=mgr.display_name_template,
                                            name_prefix="r_"))
    return env_data