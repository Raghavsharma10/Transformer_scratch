def reverse_url(self, scheme: str, path: str) -> str:
        """
        Reverses the url using scheme and path given in parameter.

        :param scheme: Scheme of the url
        :param path: Path of the url
        :return:
        """
        # remove starting slash in path if present
        path = path.lstrip('/')

        server, port = self.connection_handler.server, self.connection_handler.port
        if self.connection_handler.path:
            url = '{scheme}://{server}:{port}/{path}'.format(scheme=scheme,
                                                             server=server,
                                                             port=port,
                                                             path=path)
        else:
            url = '{scheme}://{server}:{port}/'.format(scheme=scheme,
                                                       server=server,
                                                       port=port)

        return url + path