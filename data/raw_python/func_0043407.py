def create_patch(self, patch_name):
        """Creates a patch based on the changes in the current repository. In
        case the specified patch *patch_name* already exists, ask the user to
        overwrite the patch. In case creating the patch was successful, all
        changes in the current repository are reverted. Returns ``True`` in case
        a patch was created, and ``False`` otherwise.

        :raises: :py:exc:`~stash.exception.StashException` in case *patch_name* already exists.
        """
        # Raise an exception in case the specified patch already exists.
        patch_path = self._get_patch_path(patch_name)
        if os.path.exists(patch_path):
            raise StashException("patch '%s' already exists" % patch_name)

        # Determine the contents for the new patch.
        patch = self.repository.diff()
        if patch != '':
            # Create the patch.
            patch_file = open(patch_path, 'wb')
            patch_file.write(patch.encode('utf-8'))
            patch_file.close()

            # Undo all changes in the repository, and determine which files have
            # been added or removed. Files that were added, need to be removed
            # again.
            pre_file_status = self.repository.status()
            self.repository.revert_all()
            changed_file_status = self.repository.status().difference(pre_file_status)

            # Remove all files that are created by the patch that is now being
            # stashed.
            for status, file_name in changed_file_status:
                if status == FileStatus.Added:
                    os.unlink(os.path.join(self.repository.root_path, file_name))

        # Return whether a non-empty patch was created.
        return patch != ''