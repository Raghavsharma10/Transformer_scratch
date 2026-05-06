def download_async(self, remote_path, local_path, callback=None):
        """Downloads remote resources from WebDAV server asynchronously

        :param remote_path: the path to remote resource on WebDAV server. Can be file and directory.
        :param local_path: the path to save resource locally.
        :param callback: the callback which will be invoked when downloading is complete.
        """
        target = (lambda: self.download_sync(local_path=local_path, remote_path=remote_path, callback=callback))
        threading.Thread(target=target).start()