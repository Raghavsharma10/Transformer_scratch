def activate(specifier):
    """Make a compatible version of pip importable. Raise a RuntimeError if we
    couldn't."""
    try:
        for distro in require(specifier):
            distro.activate()
    except (VersionConflict, DistributionNotFound):
        raise RuntimeError('The installed version of pip is too old; peep '
                           'requires ' + specifier)