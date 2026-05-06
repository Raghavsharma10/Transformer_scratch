def get_view_function(module_name, function_name, fallback_app=None, fallback_template=None, verify_decorator=True):
    '''
    Retrieves a view function from the cache, finding it if the first time.
    Raises ViewDoesNotExist if not found.  This is called by resolver.py.
    '''
    # first check the cache (without doing locks)
    key = ( module_name, function_name )
    try:
        return CACHED_VIEW_FUNCTIONS[key]
    except KeyError:
        with rlock:
            # try again now that we're locked
            try:
                return CACHED_VIEW_FUNCTIONS[key]
            except KeyError:
                # if we get here, we need to load the view function
                func = find_view_function(module_name, function_name, fallback_app, fallback_template, verify_decorator)
                # cache in production mode
                if not settings.DEBUG:
                    CACHED_VIEW_FUNCTIONS[key] = func
                return func

    # the code should never be able to get here
    raise Exception("Django-Mako-Plus error: get_view_function() should not have been able to get to this point.  Please notify the owner of the DMP project.  Thanks.")