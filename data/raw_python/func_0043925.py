def add_files(self, *filenames, **kw):
        """
        Include added and/or removed files in the working tree in the next commit.

        :param filenames: The filenames of the files to include in the next
                          commit (zero or more strings). If no arguments are
                          given all untracked files are added.
        :param kw: Keyword arguments are ignored (instead of raising
                   :exc:`~exceptions.TypeError`) to enable backwards
                   compatibility with older versions of `vcs-repo-mgr`
                   where the keyword argument `all` was used.
        """
        # Make sure the local repository exists and supports a working tree.
        self.create()
        self.ensure_working_tree()
        # Include added and/or removed files in the next commit.
        logger.info("Staging changes to be committed in %s ..", format_path(self.local))
        self.context.execute(*self.get_add_files_command(*filenames))