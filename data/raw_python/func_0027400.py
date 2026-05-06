def request_xml(url, auth=None):
    '''
    Returns an etree.XMLRoot object loaded from the url
    :param str url: URL for the resource to load as an XML
    '''
    try:
        r = requests.get(url, auth=auth, verify=False)
        return r.text.encode('utf-8')
    except BaseException:
        logger.error("Skipping %s (error parsing the XML)" % url)
    return