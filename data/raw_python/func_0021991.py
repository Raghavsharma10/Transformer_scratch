def _parse_authorization(cls, response, uri=None):
        """
        Parse an authorization resource.
        """
        links = _parse_header_links(response)
        try:
            new_cert_uri = links[u'next'][u'url']
        except KeyError:
            raise errors.ClientError('"next" link missing')
        return (
            response.json()
            .addCallback(
                lambda body: messages.AuthorizationResource(
                    body=messages.Authorization.from_json(body),
                    uri=cls._maybe_location(response, uri=uri),
                    new_cert_uri=new_cert_uri))
            )