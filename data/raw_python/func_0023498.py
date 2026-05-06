def download_sync(self, remote_path, local_path, callback=None):
        """Downloads remote resources from WebDAV server synchronously.

        :param remote_path: the path to remote resource on WebDAV server. Can be file and directory.
        :param local_path: the path to save resource locally.
        :param callback: the callback which will be invoked when downloading is complete.
        """
        self.download(local_path=local_path, remote_path=remote_path)
        if callback:
            callback()