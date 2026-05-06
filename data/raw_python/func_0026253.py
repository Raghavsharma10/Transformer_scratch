def semver(version, loose):
    if isinstance(version, SemVer):
        if version.loose == loose:
            return version
        else:
            version = version.version
    elif not isinstance(version, str):  # xxx:
        raise InvalidTypeIncluded("must be str, but {!r}".format(version))

    """
    if (!(this instanceof SemVer))
       return new SemVer(version, loose);
    """
    return SemVer(version, loose)