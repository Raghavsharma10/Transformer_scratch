def conn_handler(self, session: ClientSession, proxy: str = None) -> ConnectionHandler:
        """
        Return connection handler instance for the endpoint

        :param session: AIOHTTP client session instance
        :param proxy: Proxy url
        :return:
        """
        if self.server:
            return ConnectionHandler("https", "wss", self.server, self.port, self.path, session, proxy)
        elif self.ipv6:
            return ConnectionHandler("https", "wss", "[{0}]".format(self.ipv6), self.port, self.path, session, proxy)

        return ConnectionHandler("https", "wss", self.ipv4, self.port, self.path, session, proxy)