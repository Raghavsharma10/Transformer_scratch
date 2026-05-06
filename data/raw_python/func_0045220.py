def add_alt(self, alt_id, description):
        """
        Add a alternative allele format field line to the header.

        Arguments:
            alt_id (str): The id of the alternative line
            description (str): A description of the info line

        """
        alt_line = '##ALT=<ID={0},Description="{1}">'.format(
            alt_id, description
        )
        logger.info("Adding alternative allele line to vcf: {0}".format(alt_line))
        self.parse_meta_data(alt_line)
        return