def upload_async(self, remote_path, local_path, callback=None):
        """Uploads resource to remote path on WebDAV server asynchronously.
        In case resource is directory it will upload all nested files and directories.

        :param remote_path: the path for uploading resources on WebDAV server. Can be file and directory.
        :param local_path: the path to local resource for uploading.
        :param callback: the callback which will be invoked when downloading is complete.
        """
        target = (lambda: self.upload_sync(local_path=local_path, remote_path=remote_path, callback=callback))
        threading.Thread(target=target).start()