def create_branch_and_checkout(self, branch_name: str):
        """
        Creates a new branch if it doesn't exist

        Args:
            branch_name: branch name
        """
        self.create_branch(branch_name)
        self.checkout(branch_name)