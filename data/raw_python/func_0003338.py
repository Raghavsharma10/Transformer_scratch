def routeargs(path, host = None, vhost = None, method = [b'POST'], **kwargs):
        "For extra arguments, see Dispatcher.routeargs. They must be specified by keyword arguments"
        def decorator(func):
            func.routemode = 'routeargs'
            func.route_path = path
            func.route_host = host
            func.route_vhost = vhost
            func.route_method = method
            func.route_kwargs = kwargs
            return func
        return decorator