def ensure_working_tree(self):
        """
        Make sure the local repository has working tree support.

        :raises: :exc:`~vcs_repo_mgr.exceptions.MissingWorkingTreeError` when
                 the local repository doesn't support a working tree.
        """
        if not self.supports_working_tree:
            raise MissingWorkingTreeError(compact("""
                A working tree is required but the local {friendly_name}
                repository at {directory} doesn't support a working tree!
            """, friendly_name=self.friendly_name, directory=format_path(self.local)))