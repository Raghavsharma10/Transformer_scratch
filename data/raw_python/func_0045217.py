def add_meta_line(self, key, value):
        """
        Adds an arbitrary metadata line to the header.

        This must be a key value pair

        Arguments:
            key (str): The key of the metadata line
            value (str): The value of the metadata line

        """
        meta_line = '##{0}={1}'.format(
            key, value
        )
        logger.info("Adding meta line to vcf: {0}".format(meta_line))
        self.parse_meta_data(meta_line)
        return