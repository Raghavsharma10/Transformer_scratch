def _match_modes(self, remote_path, l_st):
        """Match mod, utime and uid/gid with locals one."""
        self.sftp.chmod(remote_path, S_IMODE(l_st.st_mode))
        self.sftp.utime(remote_path, (l_st.st_atime, l_st.st_mtime))

        if self.chown:
            self.sftp.chown(remote_path, l_st.st_uid, l_st.st_gid)