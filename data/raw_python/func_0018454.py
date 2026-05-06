def check_for_deletion(self, relative_path=None):
        """Traverse the entire remote_path tree.

        Find files/directories that need to be deleted,
        not being present in the local folder.
        """
        if not relative_path:
            relative_path = str()  # root of shared directory tree

        remote_path = path_join(self.remote_path, relative_path)
        local_path = path_join(self.local_path, relative_path)

        for remote_st in self.sftp.listdir_attr(remote_path):
            r_lstat = self.sftp.lstat(path_join(remote_path, remote_st.filename))

            inner_remote_path = path_join(remote_path, remote_st.filename)
            inner_local_path = path_join(local_path, remote_st.filename)

            # check if remote_st is a symlink
            # otherwise could delete file outside shared directory
            if S_ISLNK(r_lstat.st_mode):
                if self._must_be_deleted(inner_local_path, r_lstat):
                    self.remote_delete(inner_remote_path, r_lstat)
                continue

            if self._must_be_deleted(inner_local_path, remote_st):
                self.remote_delete(inner_remote_path, remote_st)
            elif S_ISDIR(remote_st.st_mode):
                self.check_for_deletion(
                    path_join(relative_path, remote_st.filename)
                )