def upload_directory(self, remote_path, local_path, progress=None):
        """Uploads directory to remote path on WebDAV server.
        In case directory is exist on remote server it will delete it and then upload directory with nested files and
        directories.

        :param remote_path: the path to directory for uploading on WebDAV server.
        :param local_path: the path to local directory for uploading.
        :param progress: Progress function. Not supported now.
        """
        urn = Urn(remote_path, directory=True)
        if not urn.is_dir():
            raise OptionNotValid(name='remote_path', value=remote_path)

        if not os.path.isdir(local_path):
            raise OptionNotValid(name='local_path', value=local_path)

        if not os.path.exists(local_path):
            raise LocalResourceNotFound(local_path)

        if self.check(urn.path()):
            self.clean(urn.path())

        self.mkdir(remote_path)

        for resource_name in listdir(local_path):
            _remote_path = f'{urn.path()}{resource_name}'
            _local_path = os.path.join(local_path, resource_name)
            self.upload(local_path=_local_path, remote_path=_remote_path, progress=progress)