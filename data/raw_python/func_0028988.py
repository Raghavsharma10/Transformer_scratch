def get_package_versions(package):
    """Get the package version information (=SetuptoolsVersion) which is
    comparable.
    note: we use the pip list_command implementation for this

    :param package: name of the package
    :return: installed version, latest available version
    """
    list_command = ListCommand()
    options, args = list_command.parse_args([])
    packages = [get_dist(package)]
    dists = list_command.iter_packages_latest_infos(packages, options)
    try:
        dist = next(dists)
        return dist.parsed_version, dist.latest_version
    except StopIteration:
        return None, None