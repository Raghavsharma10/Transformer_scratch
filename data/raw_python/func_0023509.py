def info(self, remote_path):
        """Gets information about resource on WebDAV.
        More information you can find by link http://webdav.org/specs/rfc4918.html#METHOD_PROPFIND

        :param remote_path: the path to remote resource.
        :return: a dictionary of information attributes and them values with following keys:
                 `created`: date of resource creation,
                 `name`: name of resource,
                 `size`: size of resource,
                 `modified`: date of resource modification.
        """
        urn = Urn(remote_path)
        if not self.check(urn.path()) and not self.check(Urn(remote_path, directory=True).path()):
            raise RemoteResourceNotFound(remote_path)

        response = self.execute_request(action='info', path=urn.quote())
        path = self.get_full_path(urn)
        return WebDavXmlUtils.parse_info_response(content=response.content, path=path, hostname=self.webdav.hostname)