def generate_client_callers(spec, timeout, error_callback, local, app):
    """Return a dict mapping method names to anonymous functions that
    will call the server's endpoint of the corresponding name as
    described in the api defined by the swagger dict and bravado spec"""

    callers_dict = {}

    def mycallback(endpoint):
        if not endpoint.handler_client:
            return

        callers_dict[endpoint.handler_client] = _generate_client_caller(spec, endpoint, timeout, error_callback, local, app)

    spec.call_on_each_endpoint(mycallback)

    return callers_dict