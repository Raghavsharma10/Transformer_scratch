def find_env_paths_in_basedirs(base_dirs):
    """Returns all potential envs in a basedir"""
    # get potential env path in the base_dirs
    env_path = []
    for base_dir in base_dirs:
        env_path.extend(glob.glob(os.path.join(
            os.path.expanduser(base_dir), '*', '')))
    # self.log.info("Found the following kernels from config: %s", ", ".join(venvs))

    return env_path