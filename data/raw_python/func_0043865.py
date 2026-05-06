def is_bare(self):
        """
        :data:`True` if the repository has no working tree, :data:`False` if it does.

        The value of this property is computed by checking whether the
        ``.bzr/checkout`` directory exists (it doesn't exist in Bazaar
        repositories created using ``bzr branch --no-tree ...``).
        """
        # Make sure the local repository exists.
        self.create()
        # Check the existence of the directory.
        checkout_directory = os.path.join(self.vcs_directory, 'checkout')
        return not self.context.is_directory(checkout_directory)