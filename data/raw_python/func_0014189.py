def find_view_function(module_name, function_name, fallback_app=None, fallback_template=None, verify_decorator=True):
    '''
    Finds a view function, class-based view, or template view.
    Raises ViewDoesNotExist if not found.
    '''
    dmp = apps.get_app_config('django_mako_plus')

    # I'm first calling find_spec first here beacuse I don't want import_module in
    # a try/except -- there are lots of reasons that importing can fail, and I just want to
    # know whether the file actually exists.  find_spec raises AttributeError if not found.
    try:
        spec = find_spec(module_name)
    except ValueError:
        spec = None
    if spec is None:
        # no view module, so create a view function that directly renders the template
        try:
            return create_view_for_template(fallback_app, fallback_template)
        except TemplateDoesNotExist as e:
            raise ViewDoesNotExist('view module {} not found, and fallback template {} could not be loaded ({})'.format(module_name, fallback_template, e))

    # load the module and function
    try:
        module = import_module(module_name)
        func = getattr(module, function_name)
        func.view_type = 'function'
    except ImportError as e:
        raise ViewDoesNotExist('module "{}" could not be imported: {}'.format(module_name, e))
    except AttributeError as e:
        raise ViewDoesNotExist('module "{}" found successfully, but "{}" was not found: {}'.format(module_name, function_name, e))

    # if class-based view, call as_view() to get a view function to it
    if inspect.isclass(func) and issubclass(func, View):
        func = func.as_view()
        func.view_type = 'class'

    # if regular view function, check the decorator
    elif verify_decorator and not view_function.is_decorated(func):
        raise ViewDoesNotExist("view {}.{} was found successfully, but it must be decorated with @view_function or be a subclass of django.views.generic.View.".format(module_name, function_name))

    # attach a converter to the view function
    if dmp.options['PARAMETER_CONVERTER'] is not None:
        try:
            converter = import_qualified(dmp.options['PARAMETER_CONVERTER'])(func)
            setattr(func, CONVERTER_ATTRIBUTE_NAME, converter)
        except ImportError as e:
            raise ImproperlyConfigured('Cannot find PARAMETER_CONVERTER: {}'.format(str(e)))

    # return the function/class
    return func