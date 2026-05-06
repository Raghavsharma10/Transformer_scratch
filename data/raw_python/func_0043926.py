def checkout(self, revision=None, clean=False):
        """
        Update the working tree of the local repository to the specified revision.

        :param revision: The revision to check out (a string,
                         defaults to :attr:`default_revision`).
        :param clean: :data:`True` to discard changes in the working tree,
                      :data:`False` otherwise.
        """
        # Make sure the local repository exists and supports a working tree.
        self.create()
        self.ensure_working_tree()
        # Update the working tree of the local repository.
        revision = revision or self.default_revision
        logger.info("Checking out revision '%s' in %s ..", revision, format_path(self.local))
        self.context.execute(*self.get_checkout_command(revision, clean))