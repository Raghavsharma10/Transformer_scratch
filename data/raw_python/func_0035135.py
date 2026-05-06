def scgi_request(url, methodname, *params, **kw):
    """ Send a XMLRPC request over SCGI to the given URL.

        @param url: Endpoint URL.
        @param methodname: XMLRPC method name.
        @param params: Tuple of simple python objects.
        @keyword deserialize: Parse XML result? (default is True)
        @return: XMLRPC response, or the equivalent Python data.
    """
    xmlreq = xmlrpclib.dumps(params, methodname)
    xmlresp = SCGIRequest(url).send(xmlreq)

    if kw.get("deserialize", True):
        # This fixes a bug with the Python xmlrpclib module
        # (has no handler for <i8> in some versions)
        xmlresp = xmlresp.replace("<i8>", "<i4>").replace("</i8>", "</i4>")

        # Return deserialized data
        return xmlrpclib.loads(xmlresp)[0][0]
    else:
        # Return raw XML
        return xmlresp