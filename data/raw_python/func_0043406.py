def apply_patch(self, patch_name):
        """Applies the patch *patch_name* on to the current working directory in
        case the patch exists. In case applying the patch was successful, the
        patch is automatically removed from the stash. Returns ``True`` in case
        applying the patch was successful, otherwise ``False`` is returned.

        :raises: :py:exc:`~stash.exception.StashException` in case *patch_name* does not exist.
        """
        if patch_name in self.get_patches():
            patch_path = self._get_patch_path(patch_name)

            # Apply the patch, and determine the files that have been added and
            # removed.
            pre_file_status = self.repository.status()
            patch_return_code = self.repository.apply_patch(patch_path)
            changed_file_status = self.repository.status().difference(pre_file_status)

            # Determine all files that have been added.
            for status, file_name in changed_file_status:
                if status == FileStatus.Added:
                    self.repository.add([file_name])
                elif status == FileStatus.Removed:
                    self.repository.remove([file_name])

            if patch_return_code == 0:
                # Applying the patch succeeded, remove stashed patch.
                os.unlink(patch_path)

            return patch_return_code == 0
        else:
            raise StashException("patch '%s' does not exist" % patch_name)