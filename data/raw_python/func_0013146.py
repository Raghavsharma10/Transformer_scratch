def _find_conda_env_paths_from_conda(mgr):
    """Returns a list of path as given by `conda env list --json`.

    Returns empty list, if conda couldn't be called.
    """
    # this is expensive, so make it configureable...
    if not mgr.use_conda_directly:
        return []
    mgr.log.debug("Looking for conda environments by calling conda directly...")
    import subprocess
    import json
    try:
        p = subprocess.Popen(
            ['conda', 'env', 'list', '--json'],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE)
        comm = p.communicate()
        output = comm[0].decode()
        if p.returncode != 0 or len(output) == 0:
            mgr.log.error(
                "Couldn't call 'conda' to get the environments. "
                "Output:\n%s", str(comm))
            return []
    except FileNotFoundError:
        mgr.log.error("'conda' not found in path.")
        return []
    output = json.loads(output)
    envs = output["envs"]
    # self.log.info("Found the following kernels from conda: %s", ", ".join(envs))
    return envs