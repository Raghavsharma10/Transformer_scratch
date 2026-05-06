def poll(self, force_rescan = False):
      """A generator producing (path, line) tuples with lines seen since the last time poll() was called. Will not block. Checks for new/deleted/rotated files every `interval` seconds, but will check every time if `force_rescan` is True. (default False)"""
      # Check for new, deleted, and rotated files.
      if force_rescan or time.time() > self._last_scan + self._interval:
         self._rescan(skip_to_end = False)
         self._last_scan = time.time()

      filereaders = {}
      for path, tailedfile in self._tailedfiles.iteritems():
         filereaders[path] = tailedfile.readlines()

      # One line is read from each file in turn, in an attempt to read
      # from all files evenly. They'll be in an undefined order because
      # of using a dict for filereaders, but that's not a problem
      # because some entropy here is desirable for evenness.
      while len(filereaders) > 0:
         for path in filereaders.keys():
            lines = filereaders[path]
            try:
               line, offset = lines.next()
            except StopIteration:
               # Reached end the of this file.
               del filereaders[path]
               break
            
            yield (path, offset), line