def spawn_server_api(api_name, app, api_spec, error_callback, decorator):
    """Take a a Flask app and a swagger file in YAML format describing a REST
    API, and populate the app with routes handling all the paths and methods
    declared in the swagger file.

    Also handle marshaling and unmarshaling between json and object instances
    representing the definitions from the swagger file.
    """

    def mycallback(endpoint):
        handler_func = get_function(endpoint.handler_server)

        # Generate api endpoint around that handler
        handler_wrapper = _generate_handler_wrapper(api_name, api_spec, endpoint, handler_func, error_callback, decorator)

        # Bind handler to the API path
        log.info("Binding %s %s ==> %s" % (endpoint.method, endpoint.path, endpoint.handler_server))
        endpoint_name = '_'.join([endpoint.method, endpoint.path]).replace('/', '_')
        app.add_url_rule(endpoint.path, endpoint_name, handler_wrapper, methods=[endpoint.method])


    api_spec.call_on_each_endpoint(mycallback)

    # Add custom error handlers to the app
    add_error_handlers(app)