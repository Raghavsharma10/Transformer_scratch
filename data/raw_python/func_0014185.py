def dmp_path(regex, kwargs=None, name=None, app_name=None):
    '''
    Creates a DMP-style, convention-based pattern that resolves
    to various view functions based on the 'dmp_page' value.

    The following should exist as 1) regex named groups or
    2) items in the kwargs dict:
        dmp_app         Should resolve to a name in INSTALLED_APPS.
                        If missing, defaults to DEFAULT_APP.
        dmp_page        The page name, which should resolve to a module:
                        project_dir/{dmp_app}/views/{dmp_page}.py
                        If missing, defaults to DEFAULT_PAGE.
        dmp_function    The function name (or View class name) within the module.
                        If missing, defaults to 'process_request'
        dmp_urlparams   The urlparams string to parse.
                        If missing, defaults to ''.

    The reason for this convenience function is to be similar to
    Django functions like url(), re_path(), and path().
    '''
    return PagePattern(regex, kwargs, name, app_name)