def create_release_branch(self, branch_name):
        """
        Create a new release branch.

        :param branch_name: The name of the release branch to create (a string).
        :raises: The following exceptions can be raised:

                  - :exc:`~exceptions.TypeError` when :attr:`release_scheme`
                    isn't set to 'branches'.
                  - :exc:`~exceptions.ValueError` when the branch name doesn't
                    match the configured :attr:`release_filter` or no parent
                    release branches are available.

        This method automatically checks out the new release branch, but note
        that the new branch may not actually exist until a commit has been made
        on the branch.
        """
        # Validate the release scheme.
        self.ensure_release_scheme('branches')
        # Validate the name of the release branch.
        if self.compiled_filter.match(branch_name) is None:
            msg = "The branch name '%s' doesn't match the release filter!"
            raise ValueError(msg % branch_name)
        # Make sure the local repository exists.
        self.create()
        # Figure out the correct parent release branch.
        candidates = natsort([r.revision.branch for r in self.ordered_releases] + [branch_name])
        index = candidates.index(branch_name) - 1
        if index < 0:
            msg = "Failed to determine suitable parent branch for release branch '%s'!"
            raise ValueError(msg % branch_name)
        parent_branch = candidates[index]
        self.checkout(parent_branch)
        self.create_branch(branch_name)