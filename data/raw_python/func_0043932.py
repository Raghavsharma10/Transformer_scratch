def delete_branch(self, branch_name, message=None, author=None):
        """
        Delete or close a branch in the local repository.

        :param branch_name: The name of the branch to delete or close (a string).
        :param message: The message to use when closing the branch requires a
                        commit (a string or :data:`None`, defaults to the
                        string "Closing branch NAME").
        :param author: Override :attr:`author` (refer to
                       :func:`coerce_author()` for details
                       on argument handling).
        """
        # Make sure the local repository exists.
        self.create()
        # Delete the branch in the local repository.
        logger.info("Deleting branch '%s' in %s ..", branch_name, format_path(self.local))
        self.context.execute(*self.get_delete_branch_command(
            author=(coerce_author(author) if author else self.author),
            message=(message or ("Closing branch %s" % branch_name)),
            branch_name=branch_name,
        ))