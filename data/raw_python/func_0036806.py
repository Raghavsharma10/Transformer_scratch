def _get_packages():
    # type: () -> List[Package]
    """Convert `pkg_resources.working_set` into a list of `Package` objects.

    :return: list
    """
    return [Package(pkg_obj=pkg) for pkg in sorted(pkg_resources.working_set,
                                                   key=lambda x: str(x).lower())]