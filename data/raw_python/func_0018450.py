def _must_be_deleted(local_path, r_st):
        """Return True if the remote correspondent of local_path has to be deleted.

        i.e. if it doesn't exists locally or if it has a different type from the remote one."""
        # if the file doesn't exists
        if not os.path.lexists(local_path):
            return True

        # or if the file type is different
        l_st = os.lstat(local_path)
        if S_IFMT(r_st.st_mode) != S_IFMT(l_st.st_mode):
            return True

        return False