def command(execute=None):  # noqa: E501
    """Execute a Command

    Execute a command # noqa: E501

    :param execute: The data needed to execute this command
    :type execute: dict | bytes

    :rtype: Response
    """
    if connexion.request.is_json:
        execute = Execute.from_dict(connexion.request.get_json())  # noqa: E501

    if(not hasAccess()):
        return redirectUnauthorized()

    try:
        connector = None

        parameters = {}

        if (execute.command.parameters):
            parameters = execute.command.parameters

        credentials = Credentials()
        options = Options(debug=execute.command.options['debug'], sensitive=execute.command.options['sensitive'])

        if (execute.auth):
            credentials = mapUserAuthToCredentials(execute.auth, credentials)

        if (not execute.auth.api_token):
            options.sensitive = True

        connector = Connector(options=options, credentials=credentials, command=execute.command.command,
                              parameters=parameters)

        commandHandler = connector.execute()

        response = Response(status=commandHandler.getRequest().getResponseStatusCode(),
                            body=json.loads(commandHandler.getRequest().getResponseBody()))

        if (execute.command.options['debug']):
            response.log = connector.logBuffer

        return response
    except:
        State.log.error(traceback.format_exc())
        if ('debug' in execute.command.options and execute.command.options['debug']):
            return ErrorResponse(status=500,
                                 message="Uncaught exception occured during processing. To get a larger stack trace, visit the logs.",
                                 state=traceback.format_exc(3))
        else:
            return ErrorResponse(status=500, message="")