def expand_branch_name(self, name):
        """
        Expand branch names to their unambiguous form.

        :param name: The name of a local or remote branch (a string).
        :returns: The unambiguous form of the branch name (a string).

        This internal method is used by methods like :func:`find_revision_id()`
        and :func:`find_revision_number()` to detect and expand remote branch
        names into their unambiguous form which is accepted by commands like
        ``git rev-parse`` and ``git rev-list --count``.
        """
        # If no name is given we pick the default revision.
        if not name:
            return self.default_revision
        # Run `git for-each-ref' once and remember the results.
        branches = list(self.find_branches_raw())
        # Check for an exact match against a local branch.
        for prefix, other_name, revision_id in branches:
            if prefix == 'refs/heads/' and name == other_name:
                # If we find a local branch whose name exactly matches the name
                # given by the caller then we consider the argument given by
                # the caller unambiguous.
                logger.debug("Branch name %r matches local branch.", name)
                return name
        # Check for an exact match against a remote branch.
        for prefix, other_name, revision_id in branches:
            if prefix.startswith('refs/remotes/') and name == other_name:
                # If we find a remote branch whose name exactly matches the
                # name given by the caller then we expand the name given by the
                # caller into the full %(refname) emitted by `git for-each-ref'.
                unambiguous_name = prefix + name
                logger.debug("Branch name %r matches remote branch %r.", name, unambiguous_name)
                return unambiguous_name
        # As a fall back we return the given name without expanding it.
        # This code path might not be necessary but was added out of
        # conservativeness, with the goal of trying to guarantee
        # backwards compatibility.
        logger.debug("Failed to expand branch name %r.", name)
        return name