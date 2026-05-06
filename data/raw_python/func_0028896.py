def ping(awsclient, function_name, alias_name=ALIAS_NAME, version=None):
    """Send a ping request to a lambda function.

    :param awsclient:
    :param function_name:
    :param alias_name:
    :param version:
    :return: ping response payload
    """
    log.debug('sending ping to lambda function: %s', function_name)
    payload = '{"ramuda_action": "ping"}'  # default to ping event
    # reuse invoke
    return invoke(awsclient, function_name, payload, invocation_type=None,
                  alias_name=alias_name, version=version)