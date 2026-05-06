def create_branch(self, branch_name: str):
        """
        Creates a new branch

        Args:
            branch_name: name of the branch

        """
        LOGGER.info('creating branch: %s', branch_name)
        self._validate_branch_name(branch_name)
        if branch_name in self.list_branches():
            LOGGER.error('branch already exists')
            sys.exit(-1)
        new_branch = self.repo.create_head(branch_name)
        new_branch.commit = self.repo.head.commit