def download_pom(repo_url=None, artifact=None, pom_url=None, target_dir=None):
    """
    Downloads a pom file with give GAV (as array) or from given pom_url and saves it as pom.xml into target_dir.

    :param repo_url: repository URL from which the pom should be downloaded, mandatory only if no pom_url provided
    :param artifact: MavenArtifact instance, mandatory only if no pom_url provided
    :param pom_url: URL of the pom to download, not mandatory
    :target_dir: target directory path, where the pom should be saved, not mandatory
    :returns: path to the saved pom, useful if no target_dir provided
    """
    if not pom_url:
        pom_url = urlparse.urljoin(repo_url, "%s/" % string.replace(artifact.groupId, ".", "/"))
        pom_url = urlparse.urljoin(pom_url, "%s/" % artifact.artifactId)
        pom_url = urlparse.urljoin(pom_url, "%s/" % artifact.version)
        pom_url = urlparse.urljoin(pom_url, "%s-%s.pom" % (artifact.artifactId, artifact.version))

    handler = None
    try:
        handler = urlopen(pom_url)
    except HTTPError as err:
        logging.error("Failed to download POM %s. %s", pom_url, err)
        return None

    if not target_dir:
        num = 1
        while not target_dir or os.path.exists(target_dir):
            target_dir = "/tmp/maven-temp-path-%s" % num
            num += 1

    pom_path = os.path.join(target_dir, "pom.xml")

    if handler.getcode() == 200:
        pom = handler.read()
        handler.close()
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
        pom_file = None
        try:
            pom_file = open(pom_path, "w")
            pom_file.write(pom)
        finally:
            if pom_file:
                pom_file.close()

    return pom_path