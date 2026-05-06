def upload_file(self, remote_path, local_path, progress=None):
        """Uploads file to remote path on WebDAV server. File should be 2Gb or less.
        More information you can find by link http://webdav.org/specs/rfc4918.html#METHOD_PUT

        :param remote_path: the path to uploading file on WebDAV server.
        :param local_path: the path to local file for uploading.
        :param progress: Progress function. Not supported now.
        """
        if not os.path.exists(local_path):
            raise LocalResourceNotFound(local_path)

        urn = Urn(remote_path)
        if urn.is_dir():
            raise OptionNotValid(name='remote_path', value=remote_path)

        if os.path.isdir(local_path):
            raise OptionNotValid(name='local_path', value=local_path)

        if not self.check(urn.parent()):
            raise RemoteParentNotFound(urn.path())

        with open(local_path, 'rb') as local_file:
            file_size = os.path.getsize(local_path)
            if file_size > self.large_size:
                raise ResourceTooBig(path=local_path, size=file_size, max_size=self.large_size)

            self.execute_request(action='upload', path=urn.quote(), data=local_file)