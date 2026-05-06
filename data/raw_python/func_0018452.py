def file_upload(self, local_path, remote_path, l_st):
        """Upload local_path to remote_path and set permission and mtime."""
        self.sftp.put(local_path, remote_path)
        self._match_modes(remote_path, l_st)