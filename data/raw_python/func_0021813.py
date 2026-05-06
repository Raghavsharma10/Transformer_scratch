def get_highest_version(versions):
    """
    Returns highest available version for a package in a list of versions
    Uses pkg_resources to parse the versions

    @param versions: List of PyPI package versions
    @type versions: List of strings

    @returns: string of a PyPI package version


    """
    sorted_versions = []
    for ver in versions:
        sorted_versions.append((pkg_resources.parse_version(ver), ver))

    sorted_versions = sorted(sorted_versions)
    sorted_versions.reverse()
    return sorted_versions[0][1]