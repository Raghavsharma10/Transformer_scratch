def helper_import(module_name, class_name=None):
    """
    Return class or module object.
    if the argument is only a module name and return a module object.
    if the argument is a module and class name, and return a class object.
    """
    try:
        module = __import__(module_name, globals(), locals(), [class_name])
    except (BlackbirdError, ImportError) as error:
        raise BlackbirdError(
            'can not load {0} module [{1}]'
            ''.format(module_name, str(error))
        )

    if not class_name:
        return module
    else:
        try:
            return getattr(module, class_name)
        except:
            return False