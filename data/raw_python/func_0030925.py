def make_app(global_conf, **app_conf):
    """Create a WSGI application and return it

    ``global_conf``
        The inherited configuration for this application. Normally from
        the [DEFAULT] section of the Paste ini file.

    ``app_conf``
        The application's local configuration. Normally specified in
        the [app:<name>] section of the Paste ini file (where <name>
        defaults to main).
    """
    # Configure the environment and fill conf dictionary.
    environment.load_environment(global_conf, app_conf)

    # Dispatch request to controllers.
    app = controllers.make_router()

    # Init request-dependant environment
    app = set_application_url(app)

    # CUSTOM MIDDLEWARE HERE (filtered by error handling middlewares)

    # Handle Python exceptions
    if not conf['debug']:
        def json_error_template(head_html, exception, extra):
            error_json = {
                'code': 500,
                'hint': u'See the HTTP server log to see the exception traceback.',
                'message': exception,
                }
            if head_html:
                error_json['head_html'] = head_html
            if extra:
                error_json['extra'] = extra
            return json.dumps({'error': error_json})
        weberror.errormiddleware.error_template = json_error_template
        app = weberror.errormiddleware.ErrorMiddleware(app, global_conf, **conf['errorware'])

    app = ensure_json_content_type(app)
    app = add_x_api_version_header(app)

    if conf['debug'] and ipdb is not None:
        app = launch_debugger_on_exception(app)

    return app