def conn_handler(self, session: ClientSession, proxy: str = None) -> ConnectionHandler:
        """
        Return connection handler instance for the endpoint

        :param session: AIOHTTP client session instance
        :param proxy: Proxy url
        :return:
        """
        return ConnectionHandler("https", "wss", self.server, self.port, "", session, proxy)