def _is_dir(self, f):
        '''Check if the given in-dap file is a directory'''
        return self._tar.getmember(f).type == tarfile.DIRTYPE