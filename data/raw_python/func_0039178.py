def user_registries(fs0, fs1):
    """Returns the list of user registries present on both FileSystems."""
    for user in fs0.ls('{}Users'.format(fs0.fsroot)):
        for path in user_registries_path(fs0.fsroot, user):
            if fs1.exists(path):
                yield path