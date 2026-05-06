def check_version(version):
    """Takes a version string or tuple and raises ValueError in case
    the passed version is newer than the current version of pgi.

    Keep in mind that the pgi version is different from the pygobject one.
    """

    if isinstance(version, string_types):
        version = tuple(map(int, version.split(".")))

    if version > version_info:
        str_version = ".".join(map(str, version))
        raise ValueError("pgi version '%s' requested, '%s' available" %
                         (str_version, __version__))