def get_effective_pom(pom_dir, repo_url, mvn_repo_local, profiles=None, additional_params=None):
    """
    Gets the effective pom from the downloaded pom. There has to be complete source tree (at least the pom tree) in case
    that the root pom contains some modules.

    :param pom_dir: directory where the pom is prepared (including potential patches)
    :param repo_url: repository URL, where all dependencies needed to resolve the effective POM are available
    :param mvn_repo_local: path to local repository to use if a non-default location is required
    :returns: the effective pom as a string or None if a problem occurs
    """
    global effective_pom_cache

    pom_file = None
    try:
        pom_file = open(os.path.join(pom_dir, "pom.xml"))
        pom = pom_file.read()
    finally:
        if pom_file:
            pom_file.close()
    artifact = MavenArtifact(pom=pom)
    gav = artifact.get_gav()

    eff_pom = None
    if repo_url in effective_pom_cache.keys():
        if gav in effective_pom_cache[repo_url].keys():
            if profiles in effective_pom_cache[repo_url][gav].keys():
                if additional_params in effective_pom_cache[repo_url][gav][profiles].keys():
                    eff_pom = effective_pom_cache[repo_url][gav][profiles][additional_params]

    if not eff_pom:
        try:
            eff_pom = _read_effective_pom(pom_dir, repo_url, mvn_repo_local, profiles, additional_params)
        finally:
            if eff_pom:
                effective_pom_cache.setdefault(repo_url, {}).setdefault(gav, {}).setdefault(profiles, {})[additional_params] = eff_pom

    return eff_pom