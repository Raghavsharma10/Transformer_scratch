def _convert_rename(self, fc):
        """Convert a FileRenameCommand into a new FileCommand.

        :return: None if the rename is being ignored, otherwise a
          new FileCommand based on the whether the old and new paths
          are inside or outside of the interesting locations.
          """
        old = fc.old_path
        new = fc.new_path
        keep_old = self._path_to_be_kept(old)
        keep_new = self._path_to_be_kept(new)
        if keep_old and keep_new:
            fc.old_path = self._adjust_for_new_root(old)
            fc.new_path = self._adjust_for_new_root(new)
            return fc
        elif keep_old:
            # The file has been renamed to a non-interesting location.
            # Delete it!
            old = self._adjust_for_new_root(old)
            return commands.FileDeleteCommand(old)
        elif keep_new:
            # The file has been renamed into an interesting location
            # We really ought to add it but we don't currently buffer
            # the contents of all previous files and probably never want
            # to. Maybe fast-import-info needs to be extended to
            # remember all renames and a config file can be passed
            # into here ala fast-import?
            self.warning("cannot turn rename of %s into an add of %s yet" %
                (old, new))
        return None