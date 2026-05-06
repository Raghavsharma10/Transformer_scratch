def _create_request(headers, body):
        """
        Create the SOAP 1.2 Envelope
        An ordered dictionary is required to ensure the same order is reflected in the XML, otherwise the
        SOAP Body element would appear before the Header element.
        """
        envelope = OrderedDict()
        for (namespace, alias) in Service.Namespaces.items():
            envelope['@xmlns:' + alias] = namespace
        envelope['soap:Header'] = headers
        envelope['soap:Body'] = body
        return xmltodict.unparse({'soap:Envelope': envelope}, encoding='utf-8')