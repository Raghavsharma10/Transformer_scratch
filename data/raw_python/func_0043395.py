def apply_patch(self, patch_path):
        """Applies the patch located at *patch_path*. Returns the return code of
        the patch command.
        """
        # Do not create .orig backup files, and merge files in place.
        return self._execute('patch -p1 --no-backup-if-mismatch --merge', stdout=open(os.devnull, 'w'), stdin=open(patch_path, 'r'))[0]