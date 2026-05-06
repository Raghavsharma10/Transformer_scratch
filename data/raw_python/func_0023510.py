def is_dir(self, remote_path):
        """Checks is the remote resource directory.
        More information you can find by link http://webdav.org/specs/rfc4918.html#METHOD_PROPFIND

        :param remote_path: the path to remote resource.
        :return: True in case the remote resource is directory and False otherwise.
        """
        urn = Urn(remote_path)
        parent_urn = Urn(urn.parent())
        if not self.check(urn.path()) and not self.check(Urn(remote_path, directory=True).path()):
            raise RemoteResourceNotFound(remote_path)

        response = self.execute_request(action='info', path=parent_urn.quote())
        path = self.get_full_path(urn)
        return WebDavXmlUtils.parse_is_dir_response(content=response.content, path=path, hostname=self.webdav.hostname)