def get_relative_file_path(absolute_file_path):
    '''
    Example:
    absolute_file_path = "/home/xxx/Dev/openfisca/openfisca-france/openfisca_france/param/param.xml"
    result = "openfisca_france/param/param.xml"
    '''
    global country_package_dir_path
    assert country_package_dir_path is not None
    relative_file_path = absolute_file_path[len(country_package_dir_path):]
    if relative_file_path.startswith('/'):
        relative_file_path = relative_file_path[1:]
    return relative_file_path