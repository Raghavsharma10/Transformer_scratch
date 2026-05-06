def compare_hives(fs0, fs1):
    """Compares all the windows registry hive files
    returning those which differ.

    """
    registries = []

    for path in chain(registries_path(fs0.fsroot), user_registries(fs0, fs1)):
        if fs0.checksum(path) != fs1.checksum(path):
            registries.append(path)

    return registries