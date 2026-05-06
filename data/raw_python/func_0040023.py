def system_status():  # noqa: E501
    """Retrieve the system status

    Retrieve the system status # noqa: E501


    :rtype: Response
    """
    if(not hasAccess()):
        return redirectUnauthorized()

    body = State.config.serialize(["driver", "log", "log-file", "log-colorize"])
    body.update({'debug': State.options.debug, 'sensitive': State.options.sensitive})
    return Response(status=200, body=body)