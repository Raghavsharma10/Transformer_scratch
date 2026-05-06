def get_pkglist():
    """
    Return list of all installed packages

    Note: It returns one project name per pkg no matter how many versions
    of a particular package is installed

    @returns: list of project name strings for every installed pkg

    """

    dists = Distributions()
    projects = []
    for (dist, _active) in dists.get_distributions("all"):
        if dist.project_name not in projects:
            projects.append(dist.project_name)
    return projects