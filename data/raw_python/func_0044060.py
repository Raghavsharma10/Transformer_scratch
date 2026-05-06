def _convert_copy(self, fc):
        """Convert a FileCopyCommand into a new FileCommand.

        :return: None if the copy is being ignored, otherwise a
          new FileCommand based on the whether the source and destination
          paths are inside or outside of the interesting locations.
          """
        src = fc.src_path
        dest = fc.dest_path
        keep_src = self._path_to_be_kept(src)
        keep_dest = self._path_to_be_kept(dest)
        if keep_src and keep_dest:
            fc.src_path = self._adjust_for_new_root(src)
            fc.dest_path = self._adjust_for_new_root(dest)
            return fc
        elif keep_src:
            # The file has been copied to a non-interesting location.
            # Ignore it!
            return None
        elif keep_dest:
            # The file has been copied into an interesting location
            # We really ought to add it but we don't currently buffer
            # the contents of all previous files and probably never want
            # to. Maybe fast-import-info needs to be extended to
            # remember all copies and a config file can be passed
            # into here ala fast-import?
            self.warning("cannot turn copy of %s into an add of %s yet" %
                (src, dest))
        return None