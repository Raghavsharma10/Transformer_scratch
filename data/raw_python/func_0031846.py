def _parse_response(xml):
        """
        Attempt to parse the SOAP response and return a python object
        Raise a WSManException if a Fault is found
        """
        try:
            soap_response = xmltodict.parse(xml, process_namespaces=True, namespaces=Service.Namespaces)
        except Exception:
            logging.debug('unable to parse the xml response: %s', xml)
            raise WSManException("the remote host returned an invalid soap response")

        # the delete response has an empty body
        body = soap_response['soap:Envelope']['soap:Body']
        if body is not None and 'soap:Fault' in body:
            raise WSManOperationException(body['soap:Fault']['soap:Reason']['soap:Text']['#text'])
        return body