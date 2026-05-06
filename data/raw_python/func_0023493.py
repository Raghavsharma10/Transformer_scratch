def download_from(self, buff, remote_path):
        """Downloads file from WebDAV and writes it in buffer.

        :param buff: buffer object for writing of downloaded file content.
        :param remote_path: path to file on WebDAV server.
        """
        urn = Urn(remote_path)
        if self.is_dir(urn.path()):
            raise OptionNotValid(name='remote_path', value=remote_path)

        if not self.check(urn.path()):
            raise RemoteResourceNotFound(urn.path())

        response = self.execute_request(action='download', path=urn.quote())
        buff.write(response.content)