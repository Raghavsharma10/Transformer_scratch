def get_version(package_name, ignore_cache=False):
    """ Get the version which is currently configured by the package """
    if ignore_cache:
        with microcache.temporarily_disabled():
            found = helpers.regex_in_package_file(
                VERSION_SET_REGEX, '_version.py', package_name, return_match=True
            )
    else:
        found = helpers.regex_in_package_file(
            VERSION_SET_REGEX, '_version.py', package_name, return_match=True
        )
    if found is None:
        raise ProjectError('found {}, but __version__ is not defined')
    current_version = found['version']
    return current_version