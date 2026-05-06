def get_version_storage():
    # type: () -> VersionStorage
    """ Get version storage for the given version file.

    The storage engine used depends on the extension of the *version_file*.
    """
    version_file = conf.get_path('version_file', 'VERSION')
    if version_file.endswith('.py'):
        return PyVersionStorage(version_file)
    elif version_file.endswith('package.json'):
        return NodeVersionStorage(version_file)
    else:
        return RawVersionStorage(version_file)