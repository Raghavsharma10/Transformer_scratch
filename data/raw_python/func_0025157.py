def package_has_version_file(package_name):
    """ Check to make sure _version.py is contained in the package """
    version_file_path = helpers.package_file_path('_version.py', package_name)
    return os.path.isfile(version_file_path)