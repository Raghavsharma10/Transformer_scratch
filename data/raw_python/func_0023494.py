def download(self, remote_path, local_path, progress=None):
        """Downloads remote resource from WebDAV and save it in local path.
        More information you can find by link http://webdav.org/specs/rfc4918.html#rfc.section.9.4

        :param remote_path: the path to remote resource for downloading can be file and directory.
        :param local_path: the path to save resource locally.
        :param progress: progress function. Not supported now.
        """
        urn = Urn(remote_path)
        if self.is_dir(urn.path()):
            self.download_directory(local_path=local_path, remote_path=remote_path, progress=progress)
        else:
            self.download_file(local_path=local_path, remote_path=remote_path, progress=progress)