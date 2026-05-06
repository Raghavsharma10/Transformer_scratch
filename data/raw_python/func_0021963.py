def view_name_from(path):
    "Resolve a path to the full python module name of the related view function"
    try:
        return CACHED_VIEWS[path]
        
    except KeyError:
        view = resolve(path)
        module = path
        name = ''
        if hasattr(view.func, '__module__'):
            module = resolve(path).func.__module__
        if hasattr(view.func, '__name__'):
            name = resolve(path).func.__name__
        
        view =  "%s.%s" % (module, name)
        CACHED_VIEWS[path] = view
        return view