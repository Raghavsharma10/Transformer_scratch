def upload(self, remote_path, local_path, progress=None):
        """Uploads resource to remote path on WebDAV server.
        In case resource is directory it will upload all nested files and directories.
        More information you can find by link http://webdav.org/specs/rfc4918.html#METHOD_PUT

        :param remote_path: the path for uploading resources on WebDAV server. Can be file and directory.
        :param local_path: the path to local resource for uploading.
        :param progress: Progress function. Not supported now.
        """
        if os.path.isdir(local_path):
            self.upload_directory(local_path=local_path, remote_path=remote_path, progress=progress)
        else:
            self.upload_file(local_path=local_path, remote_path=remote_path)