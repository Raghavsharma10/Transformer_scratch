def get_context_data_from_headers(request, headers_schema):
    """
    Extracts context data from request headers according to specified schema.

    >>> from lxml import etree as et
    >>> from datetime import date
    >>> from pyws.functions.args import TypeFactory
    >>> Fake = type('Fake', (object, ), {})
    >>> request = Fake()
    >>> request.parsed_data = Fake()
    >>> request.parsed_data.xml = et.fromstring(
    ...     '<s:Envelope xmlns:s="http://schemas.xmlsoap.org/soap/envelope/">'
    ...       '<s:Header>'
    ...         '<headers>'
    ...           '<string>hello</string>'
    ...           '<number>100</number>'
    ...           '<date>2011-08-12</date>'
    ...         '</headers>'
    ...       '</s:Header>'
    ...     '</s:Envelope>')
    >>> data = get_context_data_from_headers(request, TypeFactory(
    ...     {0: 'Headers', 'string': str, 'number': int, 'date': date}))
    >>> data == {'string': 'hello', 'number': 100, 'date': date(2011, 8, 12)}
    True
    """

    if not headers_schema:
        return None

    env = request.parsed_data.xml.xpath(
        '/soap:Envelope', namespaces=SoapProtocol.namespaces)[0]

    header = env.xpath(
        './soap:Header/*', namespaces=SoapProtocol.namespaces)
    if len(header) < 1:
        return None

    return headers_schema.validate(xml2obj(header[0], headers_schema))