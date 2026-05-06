def get_dist(dist_name, lookup_dirs=None):
    """Get dist for installed version of dist_name avoiding pkg_resources cache
    """
    # note: based on pip/utils/__init__.py, get_installed_version(...)

    # Create a requirement that we'll look for inside of setuptools.
    req = pkg_resources.Requirement.parse(dist_name)

    # We want to avoid having this cached, so we need to construct a new
    # working set each time.
    if lookup_dirs is None:
        working_set = pkg_resources.WorkingSet()
    else:
        working_set = pkg_resources.WorkingSet(lookup_dirs)

    # Get the installed distribution from our working set
    return working_set.find(req)