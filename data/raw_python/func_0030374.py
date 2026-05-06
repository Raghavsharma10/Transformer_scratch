def offsets(self):
      """A generator producing a (path, offset) tuple for all tailed files."""
      for path, tailedfile in self._tailedfiles.iteritems():
         yield path, tailedfile._offset