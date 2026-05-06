def _read_managed_gavs(artifact, repo_url=None, mgmt_type=MGMT_TYPE.DEPENDENCIES, mvn_repo_local=None):
    """
    Reads all artifacts managed in dependencyManagement section of effective pom of the given artifact. It places the
    repo_url in settings.xml and then runs help:effective-pom with these settings. There should be the POM with its
    parent and dependencies available in the repository and there should also be all plugins available needed to execute
    help:effective-pom goal.

    :param artifact: MavenArtifact instance representing the POM
    :param repo_url: repository URL to use
    :param mgmt_type: type of management to read, values available are defined in MGMT_TYPE class
    :param mvn_repo_local: path to local Maven repository to be used when getting effective POM
    :returns: dictionary, where key is the management type and value is the list of artifacts managed by
              dependencyManagement/pluginManagement or None, if a problem occurs
    """

    # download the pom
    pom_path = download_pom(repo_url, artifact)
    if pom_path:
        pom_dir = os.path.split(pom_path)[0]

        # get effective pom
        eff_pom = get_effective_pom(pom_dir, repo_url, mvn_repo_local)
        shutil.rmtree(pom_dir, True)
        if not eff_pom:
            return None

        # read dependencyManagement/pluginManagement section
        managed_arts = read_management(eff_pom, mgmt_type)
    else:
        managed_arts = None

    return managed_arts