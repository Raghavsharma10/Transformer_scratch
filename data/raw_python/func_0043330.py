def crab_gateway_request(client, method, *args):
    '''
    Utility function that helps making requests to the CRAB service.

    This is a specialised version of :func:`crabpy.client.crab_request` that
    allows adding extra functionality for the calls made by the gateway.

    :param client: A :class:`suds.client.Client` for the CRAB service.
    :param string action: Which method to call, eg. `ListGewesten`
    :returns: Result of the SOAP call.
    '''
    try:
        return crab_request(client, method, *args)
    except WebFault as wf:
        err = GatewayRuntimeException(
            'Could not execute request. Message from server:\n%s' % wf.fault['faultstring'],
            wf
        )
        raise err