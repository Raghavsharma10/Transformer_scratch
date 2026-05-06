def create_branch(self, branch_name):
        """
        Create a new branch based on the working tree's revision.

        :param branch_name: The name of the branch to create (a string).

        This method automatically checks out the new branch, but note that the
        new branch may not actually exist until a commit has been made on the
        branch.
        """
        # Make sure the local repository exists and supports a working tree.
        self.create()
        self.ensure_working_tree()
        # Create the new branch in the local repository.
        logger.info("Creating branch '%s' in %s ..", branch_name, format_path(self.local))
        self.context.execute(*self.get_create_branch_command(branch_name))