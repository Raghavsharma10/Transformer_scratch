def set_version(package_name, version_str):
    """ Set the version in _version.py to version_str """
    current_version = get_version(package_name)
    version_file_path = helpers.package_file_path('_version.py', package_name)
    version_file_content = helpers.get_file_content(version_file_path)
    version_file_content = version_file_content.replace(current_version, version_str)
    with open(version_file_path, 'w') as version_file:
        version_file.write(version_file_content)