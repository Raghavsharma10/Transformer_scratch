def create_tag(self, tag_name):
        """
        Create a new tag based on the working tree's revision.

        :param tag_name: The name of the tag to create (a string).
        """
        # Make sure the local repository exists and supports a working tree.
        self.create()
        self.ensure_working_tree()
        # Create the new tag in the local repository.
        logger.info("Creating tag '%s' in %s ..", tag_name, format_path(self.local))
        self.context.execute(*self.get_create_tag_command(tag_name))