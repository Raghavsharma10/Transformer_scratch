def connect_ws(self, path: str) -> _WSRequestContextManager:
        """
        Connect to a websocket in order to use API parameters

        :param path: the url path
        :return:
        """
        client = API(self.endpoint.conn_handler(self.session, self.proxy))
        return client.connect_ws(path)