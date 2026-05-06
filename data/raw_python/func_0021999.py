def _parse_certificate(cls, response):
        """
        Parse a response containing a certificate resource.
        """
        links = _parse_header_links(response)
        try:
            cert_chain_uri = links[u'up'][u'url']
        except KeyError:
            cert_chain_uri = None
        return (
            response.content()
            .addCallback(
                lambda body: messages.CertificateResource(
                    uri=cls._maybe_location(response),
                    cert_chain_uri=cert_chain_uri,
                    body=body))
            )