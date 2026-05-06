def _requirements_sanitize(req_list):
    # type: (List[str]) -> List[str]
    """
    Cleanup a list of requirement strings (e.g. from requirements.txt) to only
    contain entries valid for this platform and with the lowest required version
    only.

    Example
    -------

    >>> from sys import version_info
    >>> _requirements_sanitize([
    ...     'foo>=3.0',
    ...     "monotonic>=1.0,>0.1;python_version=='2.4'",
    ...     "bar>1.0;python_version=='{}.{}'".format(version_info[0], version_info[1])
    ... ])
    ['foo >= 3.0', 'bar > 1.0']
    """
    filtered_req_list = (
        _requirement_find_lowest_possible(req) for req in
        (pkg_resources.Requirement.parse(s) for s in req_list)
        if _requirement_filter_by_marker(req)
    )
    return [" ".join(req) for req in filtered_req_list]