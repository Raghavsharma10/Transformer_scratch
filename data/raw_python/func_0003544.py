def default_interface(ifconfig=None, route_output=None):
    """
    Return just the default interface device dictionary.

    :param ifconfig: For mocking actual command output
    :param route_output: For mocking actual command output
    """
    global Parser
    return Parser(ifconfig=ifconfig)._default_interface(route_output=route_output)