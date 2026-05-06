async def requests_get(self, path: str, **kwargs) -> ClientResponse:
        """
        Requests GET wrapper in order to use API parameters.

        :param path: the request path
        :return:
        """
        logging.debug("Request : {0}".format(self.reverse_url(self.connection_handler.http_scheme, path)))
        url = self.reverse_url(self.connection_handler.http_scheme, path)
        response = await self.connection_handler.session.get(url, params=kwargs, headers=self.headers,
                                                             proxy=self.connection_handler.proxy,
                                                             timeout=15)
        if response.status != 200:
            try:
                error_data = parse_error(await response.text())
                raise DuniterError(error_data)
            except (TypeError, jsonschema.ValidationError):
                raise ValueError('status code != 200 => %d (%s)' % (response.status, (await response.text())))

        return response