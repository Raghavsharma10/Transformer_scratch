def setup_environ(manage_file, settings=None, more_pythonic=False):
    """Sets up a Django app within a manage.py file.

    Keyword Arguments

    **settings**
        An imported settings module. Without this, playdoh tries to import
        these modules (in order): DJANGO_SETTINGS_MODULE, settings

    **more_pythonic**
        When True, does not do any path hackery besides adding the vendor dirs.
        This requires a newer Playdoh layout without top level apps, lib, etc.
    """
    # sys is global to avoid undefined local
    global sys, current_settings, execute_from_command_line, ROOT

    ROOT = os.path.dirname(os.path.abspath(manage_file))

    # Adjust the python path and put local packages in front.
    prev_sys_path = list(sys.path)

    # Make root application importable without the need for
    # python setup.py install|develop
    sys.path.append(ROOT)

    if not more_pythonic:
        warnings.warn("You're using an old-style Playdoh layout with a top "
                      "level __init__.py and apps directories. This is error "
                      "prone and fights the Zen of Python. "
                      "See http://playdoh.readthedocs.org/en/latest/"
                      "getting-started/upgrading.html")
        # Give precedence to your app's parent dir, which contains __init__.py
        sys.path.append(os.path.abspath(os.path.join(ROOT, os.pardir)))

        site.addsitedir(path('apps'))
        site.addsitedir(path('lib'))

    # Local (project) vendor library
    site.addsitedir(path('vendor-local'))
    site.addsitedir(path('vendor-local/lib/python'))

    # Global (upstream) vendor library
    site.addsitedir(path('vendor'))
    site.addsitedir(path('vendor/lib/python'))

    # Move the new items to the front of sys.path. (via virtualenv)
    new_sys_path = []
    for item in list(sys.path):
        if item not in prev_sys_path:
            new_sys_path.append(item)
            sys.path.remove(item)
    sys.path[:0] = new_sys_path

    from django.core.management import execute_from_command_line  # noqa
    if not settings:
        if 'DJANGO_SETTINGS_MODULE' in os.environ:
            settings = import_mod_by_name(os.environ['DJANGO_SETTINGS_MODULE'])
        elif os.path.isfile(os.path.join(ROOT, 'settings_local.py')):
            import settings_local as settings
            warnings.warn("Using settings_local.py is deprecated. See "
                          "http://playdoh.readthedocs.org/en/latest/upgrading.html",
                          DeprecationWarning)
        else:
            import settings
    current_settings = settings
    validate_settings(settings)