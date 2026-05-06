def connect_ws(self, path: str) -> _WSRequestContextManager:
        """
        Connect to a websocket in order to use API parameters

        In reality, aiohttp.session.ws_connect returns a aiohttp.client._WSRequestContextManager instance.
        It must be used in a with statement to get the ClientWebSocketResponse instance from it (__aenter__).
        At the end of the with statement, aiohttp.client._WSRequestContextManager.__aexit__ is called
        and close the ClientWebSocketResponse in it.

        :param path: the url path
        :return:
        """
        url = self.reverse_url(self.connection_handler.ws_scheme, path)
        return self.connection_handler.session.ws_connect(url, proxy=self.connection_handler.proxy)