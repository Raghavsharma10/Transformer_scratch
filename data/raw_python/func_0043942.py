def is_feature_branch(self, branch_name):
        """
        Try to determine whether a branch name refers to a feature branch.

        :param branch_name: The name of a branch (a string).
        :returns: :data:`True` if the branch name appears to refer to a feature
                  branch, :data:`False` otherwise.

        This method is used by :func:`merge_up()` to determine whether the
        feature branch that was merged should be deleted or closed.

        If the branch name matches :attr:`default_revision` or one of the
        branch names of the :attr:`releases` then it is not considered a
        feature branch, which means it won't be closed.
        """
        # The following checks are intentionally ordered from lightweight to heavyweight.
        if branch_name == self.default_revision:
            # The default branch is never a feature branch.
            return False
        elif branch_name not in self.branches:
            # Invalid branch names can't be feature branch names.
            return False
        elif self.release_scheme == 'branches' and branch_name in self.release_branches:
            # Release branches are not feature branches.
            return False
        else:
            # Other valid branches are considered feature branches.
            return True