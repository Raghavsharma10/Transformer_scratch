def parse(parse_obj, agent=None, etag=None, modified=None, inject=False):
    """Parse a subscription list and return a dict containing the results.

    :param parse_obj: A file-like object or a string containing a URL, an
        absolute or relative filename, or an XML document.
    :type parse_obj: str or file
    :param agent: User-Agent header to be sent when requesting a URL
    :type agent: str
    :param etag: The ETag header to be sent when requesting a URL.
    :type etag: str
    :param modified: The Last-Modified header to be sent when requesting a URL.
    :type modified: str or datetime.datetime

    :returns: All of the parsed information, webserver HTTP response
        headers, and any exception encountered.
    :rtype: dict

    :py:func:`~listparser.parse` is the only public function exposed by
    listparser.

    If *parse_obj* is a URL, the *agent* will identify the software
    making the request, *etag* will identify the last HTTP ETag
    header returned by the webserver, and *modified* will
    identify the last HTTP Last-Modified header returned by the
    webserver. *agent* and *etag* must be strings,
    while *modified* can be either a string or a Python
    *datetime.datetime* object.

    If *agent* is not provided, the :py:data:`~listparser.USER_AGENT` global
    variable will be used by default.
    """

    guarantees = common.SuperDict({
        'bozo': 0,
        'feeds': [],
        'lists': [],
        'opportunities': [],
        'meta': common.SuperDict(),
        'version': '',
    })
    fileobj, info = _mkfile(parse_obj, (agent or USER_AGENT), etag, modified)
    guarantees.update(info)
    if not fileobj:
        return guarantees

    handler = Handler()
    handler.harvest.update(guarantees)
    parser = xml.sax.make_parser()
    parser.setFeature(xml.sax.handler.feature_namespaces, True)
    parser.setContentHandler(handler)
    parser.setErrorHandler(handler)
    if inject:
        fileobj = Injector(fileobj)
    try:
        parser.parse(fileobj)
    except (SAXParseException, MalformedByteSequenceException):  # noqa: E501  # pragma: no cover
        # Jython propagates exceptions past the ErrorHandler.
        err = sys.exc_info()[1]
        handler.harvest.bozo = 1
        handler.harvest.bozo_exception = err
    finally:
        fileobj.close()

    # Test if a DOCTYPE injection is needed
    if hasattr(handler.harvest, 'bozo_exception'):
        if 'entity' in handler.harvest.bozo_exception.__str__():
            if not inject:
                return parse(parse_obj, agent, etag, modified, True)
    # Make it clear that the XML file is broken
    # (if no other exception has been assigned)
    if inject and not handler.harvest.bozo:
        handler.harvest.bozo = 1
        handler.harvest.bozo_exception = ListError('undefined entity found')
    return handler.harvest