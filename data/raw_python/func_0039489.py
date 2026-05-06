def klass_to_pypath(klass):
    """when a class is defined within the module that is being executed as
    main, the module name will be specified as '__main__' even though the
    module actually had its own real name.  This ends up being very confusing
    to Configman as it tries to refer to a class by its proper module name.
    This function will convert a class into its properly qualified actual
    pathname.  This method is used when a Socorro app is actually invoked
    directly through the file in which the App class is defined.  This allows
    configman to reimport the class under its proper name and treat it as if
    it had been run through the SocorroWelcomeApp.  In turn, this allows
    the application defaults to be fetched from the properly imported class
    in time for configman use that information as value source."""
    if klass.__module__ == '__main__':
        module_path = (
            sys.modules['__main__']
            .__file__[:-3]
        )
        module_name = ''
        for a_python_path in sys.path:
            tentative_pathname = module_path.replace(a_python_path, '')
            if tentative_pathname != module_path:
                module_name = (
                    tentative_pathname.replace('/', '.').strip('.')
                )
                break
        if module_name == '':
            return py_obj_to_str(klass)
    else:
        module_name = klass.__module__
    return "%s.%s" % (module_name, klass.__name__)