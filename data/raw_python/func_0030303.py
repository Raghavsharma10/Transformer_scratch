def get_scm_status(config, read_modules=False, repo_url=None, mvn_repo_local=None, additional_params=None):
    """
    Gets the artifact status (MavenArtifact instance) from SCM defined by config. Only the top-level artifact is read by
    default, although it can be requested to read the whole available module structure.

    :param config: artifact config (ArtifactConfig instance)
    :param read_modules: if True all modules are read, otherwise only top-level artifact
    :param repo_url: the URL of the repository to use
    :param mvn_repo_local: local repository path
    :param additional_params: additional params to add on command-line when running maven
    """
    global scm_status_cache
    if config.artifact in scm_status_cache.keys():
        result = scm_status_cache[config.artifact]
    elif not read_modules and (("%s|False" % config.artifact) in scm_status_cache.keys()):
        result = scm_status_cache["%s|False" % config.artifact]
    else:
        result = _get_scm_status(config, read_modules, repo_url, mvn_repo_local, additional_params)
        if read_modules:
            scm_status_cache[config.artifact] = result
            if ("%s|False" % config.artifact) in scm_status_cache.keys():
                del(scm_status_cache["%s|False" % config.artifact])
        else:
            scm_status_cache["%s|False" % config.artifact] = result
    return result