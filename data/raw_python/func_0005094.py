def write(version):
    # type: (str) -> None
    """ Write the given version to the VERSION_FILE """
    if not is_valid(version):
        raise ValueError("Invalid version: ".format(version))

    storage = get_version_storage()
    storage.write(version)