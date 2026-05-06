def crab_request(client, action, *args):
    '''
    Utility function that helps making requests to the CRAB service.

    :param client: A :class:`suds.client.Client` for the CRAB service.
    :param string action: Which method to call, eg. `ListGewesten`
    :returns: Result of the SOAP call.

    .. versionadded:: 0.3.0
    '''
    log.debug('Calling %s on CRAB service.', action)
    return getattr(client.service, action)(*args)