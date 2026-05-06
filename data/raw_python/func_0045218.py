def add_filter(self, filter_id, description):
        """
        Add a filter line to the header.

        Arguments:
            filter_id (str): The id of the filter line
            description (str): A description of the info line

        """
        filter_line = '##FILTER=<ID={0},Description="{1}">'.format(
            filter_id, description
        )
        logger.info("Adding filter line to vcf: {0}".format(filter_line))
        self.parse_meta_data(filter_line)
        return