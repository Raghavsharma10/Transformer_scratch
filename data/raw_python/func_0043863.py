def crab_factory(**kwargs):
    '''
    Factory that generates a CRAB client.

    A few parameters will be handled by the factory, other parameters will
    be passed on to the client.

    :param wsdl: `Optional.` Allows overriding the default CRAB wsdl url.
    :param proxy: `Optional.` A dictionary of proxy information that is passed
        to the underlying :class:`suds.client.Client`
    :rtype: :class:`suds.client.Client`
    '''
    if 'wsdl' in kwargs:
        wsdl = kwargs['wsdl']
        del kwargs['wsdl']
    else:
        wsdl = "http://crab.agiv.be/wscrab/wscrab.svc?wsdl"
    log.info('Creating CRAB client with wsdl: %s', wsdl)
    c = Client(
        wsdl,
        **kwargs
    )
    return c