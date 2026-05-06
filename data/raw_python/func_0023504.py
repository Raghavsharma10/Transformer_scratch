def upload_sync(self, remote_path, local_path, callback=None):
        """Uploads resource to remote path on WebDAV server synchronously.
        In case resource is directory it will upload all nested files and directories.

        :param remote_path: the path for uploading resources on WebDAV server. Can be file and directory.
        :param local_path: the path to local resource for uploading.
        :param callback: the callback which will be invoked when downloading is complete.
        """
        self.upload(local_path=local_path, remote_path=remote_path)

        if callback:
            callback()