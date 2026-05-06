def get_patch(cls, patch_name):
        """Returns the contents of the specified patch *patch_name*.

        :raises: :py:exc:`~stash.exception.StashException` in case *patch_name* does not exist.
        """
        try:
            return open(cls._get_patch_path(patch_name), 'r').read()
        except:
            raise StashException("patch '%s' does not exist" % patch_name)