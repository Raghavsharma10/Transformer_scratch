def action(route, template='', methods=['GET']):
    """Decorator to create an action"""
    def real_decorator(function):
        function.pi_api_action = True
        function.pi_api_route = route
        function.pi_api_template = template
        function.pi_api_methods = methods

        if hasattr(function, 'pi_api_crossdomain'):
            if not function.pi_api_crossdomain_data['methods']:
                function.pi_api_crossdomain_data['methods'] = methods

            if 'OPTIONS' not in function.pi_api_methods:
                function.pi_api_methods += ['OPTIONS']

        return function
    return real_decorator