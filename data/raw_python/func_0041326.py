def send_response(self, response_dict):
        """
        Encode a response according to the request.

        :param dict response_dict: the response to send

        :raises: :class:`tornado.web.HTTPError` if no acceptable content
            type exists

        This method will encode `response_dict` using the most appropriate
        encoder based on the :mailheader:`Accept` request header and the
        available encoders.  The result is written to the client by calling
        ``self.write`` after setting the response content type using
        ``self.set_header``.

        """
        accept = headers.parse_http_accept_header(
            self.request.headers.get('Accept', '*/*'))
        try:
            selected, _ = algorithms.select_content_type(
                accept, _content_types.values())
        except errors.NoMatch:
            raise web.HTTPError(406,
                                'no acceptable content type for %s in %r',
                                accept, _content_types.values(),
                                reason='Content Type Not Acceptable')

        LOGGER.debug('selected %s as outgoing content type', selected)
        handler = _content_handlers[str(selected)]

        accept = self.request.headers.get('Accept-Charset', '*')
        charsets = headers.parse_accept_charset(accept)
        charset = charsets[0] if charsets[0] != '*' else None
        LOGGER.debug('encoding response body using %r with encoding %s',
                     handler, charset)
        encoding, response_bytes = handler.pack_bytes(response_dict,
                                                      encoding=charset)

        if encoding:  # don't overwrite the value in _content_types
            copied = datastructures.ContentType(selected.content_type,
                                                selected.content_subtype,
                                                selected.parameters)
            copied.parameters['charset'] = encoding
            selected = copied
        self.set_header('Content-Type', str(selected))
        self.write(response_bytes)