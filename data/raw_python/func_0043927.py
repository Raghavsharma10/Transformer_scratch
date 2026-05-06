def commit(self, message, author=None):
        """
        Commit changes to tracked files in the working tree.

        :param message: The commit message (a string).
        :param author: Override :attr:`author` (refer to
                       :func:`coerce_author()` for details
                       on argument handling).
        """
        # Make sure the local repository exists and supports a working tree.
        self.ensure_exists()
        self.ensure_working_tree()
        logger.info("Committing changes in %s: %s", format_path(self.local), message)
        author = coerce_author(author) if author else self.author
        self.context.execute(*self.get_commit_command(message, author))