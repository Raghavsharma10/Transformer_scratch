def _requirement_filter_by_marker(req):
    # type: (pkg_resources.Requirement) -> bool
    """Check if the requirement is satisfied by the marker.

    This function checks for a given Requirement whether its environment marker
    is satisfied on the current platform. Currently only the python version and
    system platform are checked.
    """
    if hasattr(req, 'marker') and req.marker:
        marker_env = {
            'python_version': '.'.join(map(str, sys.version_info[:2])),
            'sys_platform': sys.platform
        }
        if not req.marker.evaluate(environment=marker_env):
            return False
    return True