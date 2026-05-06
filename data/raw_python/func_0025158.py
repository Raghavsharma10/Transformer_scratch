def get_project_name():
    """ Grab the project name out of setup.py """
    setup_py_content = helpers.get_file_content('setup.py')
    ret = helpers.value_of_named_argument_in_function(
        'name', 'setup', setup_py_content, resolve_varname=True
    )
    if ret and ret[0] == ret[-1] in ('"', "'"):
        ret = ret[1:-1]
    return ret