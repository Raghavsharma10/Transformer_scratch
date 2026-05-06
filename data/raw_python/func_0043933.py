def ensure_clean(self):
        """
        Make sure the working tree is clean (contains no changes to tracked files).

        :raises: :exc:`~vcs_repo_mgr.exceptions.WorkingTreeNotCleanError`
                 when the working tree contains changes to tracked files.
        """
        if not self.is_clean:
            raise WorkingTreeNotCleanError(compact("""
                The repository's local working tree ({local})
                contains changes to tracked files!
            """, local=format_path(self.local)))