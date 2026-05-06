def fsroot(self):
        """Returns the file system root."""
        if self.osname == 'windows':
            return '{}:\\'.format(
                self._handler.inspect_get_drive_mappings(self._root)[0][0])
        else:
            return self._handler.inspect_get_mountpoints(self._root)[0][0]