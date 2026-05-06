def symlink_to(self, target, target_is_directory=False):
        """
        Make this path a symlink pointing to the given path.
        Note the order of arguments (self, target) is the reverse of os.symlink's.
        """
        self._accessor.symlink(target, self, target_is_directory)