def _parse_regr_response(self, response, uri=None, new_authzr_uri=None,
                             terms_of_service=None):
        """
        Parse a registration response from the server.
        """
        links = _parse_header_links(response)
        if u'terms-of-service' in links:
            terms_of_service = links[u'terms-of-service'][u'url']
        if u'next' in links:
            new_authzr_uri = links[u'next'][u'url']
        if new_authzr_uri is None:
            raise errors.ClientError('"next" link missing')
        return (
            response.json()
            .addCallback(
                lambda body:
                messages.RegistrationResource(
                    body=messages.Registration.from_json(body),
                    uri=self._maybe_location(response, uri=uri),
                    new_authzr_uri=new_authzr_uri,
                    terms_of_service=terms_of_service))
            )