def remote_delete(self, remote_path, r_st):
        """Remove the remote directory node."""
        # If it's a directory, then delete content and directory
        if S_ISDIR(r_st.st_mode):
            for item in self.sftp.listdir_attr(remote_path):
                full_path = path_join(remote_path, item.filename)
                self.remote_delete(full_path, item)
            self.sftp.rmdir(remote_path)

        # Or simply delete files
        else:
            try:
                self.sftp.remove(remote_path)
            except FileNotFoundError as e:
                self.logger.error(
                    "error while removing {}. trace: {}".format(remote_path, e)
                )