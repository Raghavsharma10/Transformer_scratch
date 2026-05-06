def remove_patch(cls, patch_name):
        """Removes patch *patch_name* from the stash (in case it exists).

        :raises: :py:exc:`~stash.exception.StashException` in case *patch_name* does not exist.
        """
        try:
            os.unlink(cls._get_patch_path(patch_name))
        except:
            raise StashException("patch '%s' does not exist" % patch_name)