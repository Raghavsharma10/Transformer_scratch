def _requirement_find_lowest_possible(req):
    # type: (pkg_resources.Requirement) -> List[str]
    """Find lowest required version.

    Given a single Requirement, this function calculates the lowest required
    version to satisfy it. If the requirement excludes a specific version, then
    this version will not be used as the minimal supported version.

    Examples
    --------

    >>> req = pkg_resources.Requirement.parse("foobar>=1.0,>2")
    >>> _requirement_find_lowest_possible(req)
    ['foobar', '>=', '1.0']
    >>> req = pkg_resources.Requirement.parse("baz>=1.3,>3,!=1.5")
    >>> _requirement_find_lowest_possible(req)
    ['baz', '>=', '1.3']

    """
    version_dep = None  # type: Optional[str]
    version_comp = None  # type: Optional[str]
    for dep in req.specs:
        version = pkg_resources.parse_version(dep[1])
        # we don't want to have a not supported version as minimal version
        if dep[0] == '!=':
            continue
        # try to use the lowest version available
        # i.e. for ">=0.8.4,>=0.9.7", select "0.8.4"
        if (not version_dep or
                version < pkg_resources.parse_version(version_dep)):
            version_dep = dep[1]
            version_comp = dep[0]

    assert (version_dep is None and version_comp is None) or \
        (version_dep is not None and version_comp is not None)

    return [
        x for x in (req.unsafe_name, version_comp, version_dep)
        if x is not None]