def get_script(name=None):  # noqa: E501
    """Retrieve the contents of a script

    Retrieve the contents of a script # noqa: E501

    :param name: The script name.
    :type name: str

    :rtype: Response
    """

    if(not hasAccess()):
        return redirectUnauthorized()

    driver = LoadedDrivers.getDefaultBaseDriver()
    return Response(status=200, body=driver.readScript(name))