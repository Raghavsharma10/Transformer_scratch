def is_valid_release_version(version):
    '''Checks that the given version code is valid.'''
    return version is not None and len(version) == 6 and version[0] == 'R' \
            and int(version[1:5]) in range(1990, 2050) \
            and version[5] in ('h', 'g', 'f', 'e', 'd', 'c', 'b', 'a')