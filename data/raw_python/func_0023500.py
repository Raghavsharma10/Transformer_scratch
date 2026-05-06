def upload_to(self, buff, remote_path):
        """Uploads file from buffer to remote path on WebDAV server.
        More information you can find by link http://webdav.org/specs/rfc4918.html#METHOD_PUT

        :param buff: the buffer with content for file.
        :param remote_path: the path to save file remotely on WebDAV server.
        """
        urn = Urn(remote_path)
        if urn.is_dir():
            raise OptionNotValid(name='remote_path', value=remote_path)

        if not self.check(urn.parent()):
            raise RemoteParentNotFound(urn.path())

        self.execute_request(action='upload', path=urn.quote(), data=buff)