def regex_in_package_file(regex, filename, package_name, return_match=False):
    """ Search for a regex in a file contained within the package directory

    If return_match is True, return the found object instead of a boolean
    """
    filepath = package_file_path(filename, package_name)
    return regex_in_file(regex, filepath, return_match=return_match)