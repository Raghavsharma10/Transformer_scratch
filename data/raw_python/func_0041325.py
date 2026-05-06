def get_request_body(self):
        """
        Decodes the request body and returns it.

        :return: the decoded request body as a :class:`dict` instance.
        :raises: :class:`tornado.web.HTTPError` if the body cannot be
            decoded (415) or if decoding fails (400)

        """
        if self._request_body is None:
            content_type_str = self.request.headers.get(
                'Content-Type', 'application/octet-stream')
            LOGGER.debug('decoding request body of type %s', content_type_str)
            content_type = headers.parse_content_type(content_type_str)
            try:
                selected, requested = algorithms.select_content_type(
                    [content_type], _content_types.values())
            except errors.NoMatch:
                raise web.HTTPError(
                    415, 'cannot decoded content type %s', content_type_str,
                    reason='Unexpected content type')
            handler = _content_handlers[str(selected)]
            try:
                self._request_body = handler.unpack_bytes(
                    self.request.body,
                    encoding=content_type.parameters.get('charset'),
                )
            except ValueError as error:
                raise web.HTTPError(
                    400, 'failed to decode content body - %r', error,
                    reason='Content body decode failure')
        return self._request_body