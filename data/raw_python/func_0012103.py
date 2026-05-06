def copy(self, new_fn):
        """copy the file to the new_fn, preserving atime and mtime"""
        new_file = self.__class__(fn=str(new_fn))
        new_file.write(data=self.read())
        new_file.utime(self.atime, self.mtime)
        return new_file