def merge_up(self, target_branch=None, feature_branch=None, delete=True, create=True):
        """
        Merge a change into one or more release branches and the default branch.

        :param target_branch: The name of the release branch where merging of
                              the feature branch starts (a string or
                              :data:`None`, defaults to
                              :attr:`current_branch`).
        :param feature_branch: The feature branch to merge in (any value
                               accepted by :func:`coerce_feature_branch()`).
        :param delete: :data:`True` (the default) to delete or close the
                       feature branch after it is merged, :data:`False`
                       otherwise.
        :param create: :data:`True` to automatically create the target branch
                       when it doesn't exist yet, :data:`False` otherwise.
        :returns: If `feature_branch` is given the global revision id of the
                  feature branch is returned, otherwise the global revision id
                  of the target branch (before any merges performed by
                  :func:`merge_up()`) is returned. If the target branch is
                  created by :func:`merge_up()` and `feature_branch` isn't
                  given then :data:`None` is returned.
        :raises: The following exceptions can be raised:

                 - :exc:`~exceptions.TypeError` when `target_branch` and
                   :attr:`current_branch` are both :data:`None`.
                 - :exc:`~exceptions.ValueError` when the given target branch
                   doesn't exist (based on :attr:`branches`) and `create` is
                   :data:`False`.
                 - :exc:`~executor.ExternalCommandFailed` if a command fails.
        """
        timer = Timer()
        repository_was_created = self.create()
        revision_to_merge = None
        # Default the target branch to the current branch.
        if not target_branch:
            target_branch = self.current_branch
            if not target_branch:
                raise TypeError("You need to specify the target branch! (where merging starts)")
        # Parse the feature branch specification.
        feature_branch = coerce_feature_branch(feature_branch) if feature_branch else None
        # Make sure we start with a clean working tree.
        self.ensure_clean()
        # Make sure we're up to date with our upstream repository (if any).
        if not repository_was_created:
            self.pull()
        # Checkout or create the target branch.
        logger.debug("Checking if target branch exists (%s) ..", target_branch)
        if target_branch in self.branches:
            self.checkout(revision=target_branch)
            # Get the global revision id of the release branch we're about to merge.
            revision_to_merge = self.find_revision_id(target_branch)
        elif not create:
            raise ValueError("The target branch %r doesn't exist!" % target_branch)
        elif self.compiled_filter.match(target_branch):
            self.create_release_branch(target_branch)
        else:
            self.create_branch(target_branch)
        # Check if we need to merge in a feature branch.
        if feature_branch:
            if feature_branch.location:
                # Pull in the feature branch.
                self.pull(remote=feature_branch.location,
                          revision=feature_branch.revision)
            # Get the global revision id of the feature branch we're about to merge.
            revision_to_merge = self.find_revision_id(feature_branch.revision)
            # Merge in the feature branch.
            self.merge(revision=feature_branch.revision)
            # Commit the merge.
            self.commit(message="Merged %s" % feature_branch.expression)
        # We skip merging up through release branches when the target branch is
        # the default branch (in other words, there's nothing to merge up).
        if target_branch != self.default_revision:
            # Find the release branches in the repository.
            release_branches = [release.revision.branch for release in self.ordered_releases]
            logger.debug("Found %s: %s",
                         pluralize(len(release_branches), "release branch", "release branches"),
                         concatenate(release_branches))
            # Find the release branches after the target branch.
            later_branches = release_branches[release_branches.index(target_branch) + 1:]
            logger.info("Found %s after target branch (%s): %s",
                        pluralize(len(later_branches), "release branch", "release branches"),
                        target_branch,
                        concatenate(later_branches))
            # Determine the branches that need to be merged.
            branches_to_upmerge = later_branches + [self.default_revision]
            logger.info("Merging up from '%s' to %s: %s",
                        target_branch,
                        pluralize(len(branches_to_upmerge), "branch", "branches"),
                        concatenate(branches_to_upmerge))
            # Merge the feature branch up through the selected branches.
            merge_queue = [target_branch] + branches_to_upmerge
            while len(merge_queue) >= 2:
                from_branch = merge_queue[0]
                to_branch = merge_queue[1]
                logger.info("Merging '%s' into '%s' ..", from_branch, to_branch)
                self.checkout(revision=to_branch)
                self.merge(revision=from_branch)
                self.commit(message="Merged %s" % from_branch)
                merge_queue.pop(0)
        # Check if we need to delete or close the feature branch.
        if delete and feature_branch and self.is_feature_branch(feature_branch.revision):
            # Delete or close the feature branch.
            self.delete_branch(
                branch_name=feature_branch.revision,
                message="Closing feature branch %s" % feature_branch.revision,
            )
            # Update the working tree to the default branch.
            self.checkout()
        logger.info("Done! Finished merging up in %s.", timer)
        return revision_to_merge