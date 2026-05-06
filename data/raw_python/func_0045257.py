def routing(routes, request):
    """Definition for route matching : helper"""

    # strip trailing slashes from request path
    path = request.path.strip('/')

    # iterate through routes to match
    args = {}
    for name, route in routes.items():
        if route['path'] == '^':
            # this section exists because regex doesn't work for null character as desired
            if path == '':
                match = [True]
            else:
                match = []
        else:
            match = re.findall(route['path'], path)
        
        if match:
            # found the matching url, iterate through variables to pass data
            # check if method exists
            if not request.method in route['method']:
                raise TornMethodNotAllowed
            
            values = match[0] # in form of tuples
            if type(values) != bool:
                for i in range(len(route['variables'])):
                    # if value is blank, check if default exists and pass it instead
                    if type(values) == str:
                        args[route['variables'][i]] = values
                    else:
                        if not values[i] and route['variables'][i] in route['defaults']:
                            values[i] = route['defaults'][route['variables'][i]]
                        args[route['variables'][i]] = values[i]
            
            # we have the variables we need, args, path, controller
            return {
                'kwargs'        : args,
                'controller'    : route['controller']
            }
    raise TornNotFoundError