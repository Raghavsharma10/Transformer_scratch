def hasBeenRotated(self):
      """Returns a boolean indicating whether the file has been removed and recreated during the time it has been open."""
      try:
         # If the inodes don't match, it means the file has been replaced.
         # The inode number cannot be recycled as long as we hold the
         # filehandle open, so this test can be trusted.
         return os.stat(self._path).st_ino != self._inode
      except OSError:
         # If the file doesn't exist, let's call it "rotated".
         return True