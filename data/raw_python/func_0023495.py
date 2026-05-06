def download_directory(self, remote_path, local_path, progress=None):
        """Downloads directory and downloads all nested files and directories from remote WebDAV to local.
        If there is something on local path it deletes directories and files then creates new.

        :param remote_path: the path to directory for downloading form WebDAV server.
        :param local_path: the path to local directory for saving downloaded files and directories.
        :param progress: Progress function. Not supported now.
        """
        urn = Urn(remote_path, directory=True)
        if not self.is_dir(urn.path()):
            raise OptionNotValid(name='remote_path', value=remote_path)

        if os.path.exists(local_path):
            shutil.rmtree(local_path)

        os.makedirs(local_path)

        for resource_name in self.list(urn.path()):
            _remote_path = f'{urn.path()}{resource_name}'
            _local_path = os.path.join(local_path, resource_name)
            self.download(local_path=_local_path, remote_path=_remote_path, progress=progress)