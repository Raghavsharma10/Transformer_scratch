def get_download_uri(package_name, version, source, index_url=None):

    """
    Use setuptools to search for a package's URI

    @returns: URI string
    """
    tmpdir = None
    force_scan = True
    develop_ok = False
    if not index_url:
        index_url = 'http://cheeseshop.python.org/pypi'

    if version:
        pkg_spec = "%s==%s" % (package_name, version)
    else:
        pkg_spec = package_name
    req = pkg_resources.Requirement.parse(pkg_spec)
    pkg_index = MyPackageIndex(index_url)
    try:
        pkg_index.fetch_distribution(req, tmpdir, force_scan, source,
                develop_ok)
    except DownloadURI as url:
        #Remove #egg=pkg-dev
        clean_url = url.value.split("#")[0]
        #If setuptools is asked for an egg and there isn't one, it will
        #return source if available, which we don't want.
        if not source and not clean_url.endswith(".egg") and \
                not clean_url.endswith(".EGG"):
            return
        else:
            return clean_url