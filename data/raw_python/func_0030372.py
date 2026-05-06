def _rescan(self, skip_to_end = True):
      """Check for new files, deleted files, and rotated files."""
      # Get listing of matching files.
      paths = []
      for single_glob in self._globspec:
          paths.extend(glob.glob(single_glob))

      # Remove files that don't appear in the new list.
      for path in self._tailedfiles.keys():
         if path not in paths:
            self._tailedfiles[path]._close()
            del self._tailedfiles[path]

      # Add any files we don't have open yet.
      for path in paths:
         try:
            # If the file has been rotated, reopen it.
            if self._tailedfiles[path].hasBeenRotated():
               # If it can't be reopened, close it.
               if not self._tailedfiles[path].reopen():
                  del self._tailedfiles[path]
         except KeyError:
            # Open a file that we haven't seen yet.
            self._tailedfiles[path] = TailedFile(path, skip_to_end = skip_to_end, offset = self._offsets.get(path, None))