def task_list():
    """
    Scans the modules set in RQ_JOBS_MODULES for RQ jobs decorated with @task
    Compiles a readable list for Job model task choices
    """
    try:
        jobs_module = settings.RQ_JOBS_MODULE
    except AttributeError:
        raise ImproperlyConfigured(_("You have to define RQ_JOBS_MODULE in settings.py"))

    if isinstance(jobs_module, string_types):
        jobs_modules = (jobs_module,)
    elif isinstance(jobs_module, (tuple, list)):
        jobs_modules = jobs_module
    else:
        raise ImproperlyConfigured(_("RQ_JOBS_MODULE must be a string or a tuple"))

    choices = []

    for module in jobs_modules:
        try:
            tasks = importlib.import_module(module)
        except ImportError:
            raise ImproperlyConfigured(_("Can not find module {}").format(module))

        module_choices = [('%s.%s' % (module, x), underscore_to_camelcase(x)) for x, y in list(tasks.__dict__.items())
                          if type(y) == FunctionType and hasattr(y, 'delay')]

        choices.extend(module_choices)

    choices.sort(key=lambda tup: tup[1])

    return choices