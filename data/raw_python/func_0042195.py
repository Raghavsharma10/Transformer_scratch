def get_filtered_register_graph(register_uri, g):
    """
    Gets a filtered version (label, comment, contained item classes & subregisters only) of the each register for the
    Register of Registers

    :param register_uri: the public URI of the register
    :type register_uri: string
    :param g: the rdf graph to append registers to
    :type g: Graph
    :return: True if ok, else False
    :rtype: boolean
    """
    import requests
    from pyldapi.exceptions import ViewsFormatsException
    assert isinstance(g, Graph)
    logging.debug('assessing register candidate ' + register_uri.replace('?_view=reg&_format=text/turtle', ''))
    try:
        r = requests.get(register_uri)
        print('getting ' + register_uri)
    except ViewsFormatsException as e:
        return False  # ignore these exceptions as are just a result of requesting a view/format combo of something like a page
    if r.status_code == 200:
        return _filter_register_graph(register_uri.replace('?_view=reg&_format=text/turtle', ''), r, g)
    logging.debug('{} returns no HTTP 200'.format(register_uri))
    return False