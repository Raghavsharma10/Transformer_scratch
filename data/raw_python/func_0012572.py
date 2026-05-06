def fix_module(job):
    """
    Fix for tasks without a module. Provides backwards compatibility with < 0.1.5
    """
    modules = settings.RQ_JOBS_MODULE
    if not type(modules) == tuple:
        modules = [modules]
    for module in modules:
        try:
            module_match = importlib.import_module(module)
            if hasattr(module_match, job.task):
                job.task = '{}.{}'.format(module, job.task)
                break
        except ImportError:
            continue
    return job