def merge(self, revision=None):
        """
        Merge a revision into the current branch (without committing the result).

        :param revision: The revision to merge in (a string or :data:`None`,
                         defaults to :attr:`default_revision`).
        :raises: The following exceptions can be raised:

                 - :exc:`~vcs_repo_mgr.exceptions.MergeConflictError` if the
                   merge command reports an error and merge conflicts are
                   detected that can't be (or haven't been) resolved
                   interactively.
                 - :exc:`~executor.ExternalCommandFailed` if the merge command
                   reports an error but no merge conflicts are detected.

        Refer to the documentation of :attr:`merge_conflict_handler` if you
        want to customize the handling of merge conflicts.
        """
        # Make sure the local repository exists and supports a working tree.
        self.create()
        self.ensure_working_tree()
        # Merge the specified revision into the current branch.
        revision = revision or self.default_revision
        logger.info("Merging revision '%s' in %s ..", revision, format_path(self.local))
        try:
            self.context.execute(*self.get_merge_command(revision))
        except ExternalCommandFailed as e:
            # Check for merge conflicts.
            conflicts = self.merge_conflicts
            if conflicts:
                # Always warn about merge conflicts and log the relevant filenames.
                explanation = format("Merge failed due to conflicts in %s! (%s)",
                                     pluralize(len(conflicts), "file"),
                                     concatenate(sorted(conflicts)))
                logger.warning("%s", explanation)
                if self.merge_conflict_handler(e):
                    # Trust the operator (or caller) and swallow the exception.
                    return
                else:
                    # Raise a specific exception for merge conflicts.
                    raise MergeConflictError(explanation)
            else:
                # Don't swallow the exception or obscure the traceback
                # in case we're not `allowed' to handle the exception.
                raise