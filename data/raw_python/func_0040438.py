async def requests_post(self, path: str, **kwargs) -> ClientResponse:
        """
        Requests POST wrapper in order to use API parameters.

        :param path: the request path
        :return:
        """
        if 'self_' in kwargs:
            kwargs['self'] = kwargs.pop('self_')

        logging.debug("POST : {0}".format(kwargs))
        response = await self.connection_handler.session.post(
            self.reverse_url(self.connection_handler.http_scheme, path),
            data=kwargs,
            headers=self.headers,
            proxy=self.connection_handler.proxy,
            timeout=15
        )
        return response