def add_contig(self, contig_id, length):
        """
        Add a contig line to the header.

        Arguments:
            contig_id (str): The id of the alternative line
            length (str): A description of the info line

        """
        contig_line = '##contig=<ID={0},length={1}>'.format(
            contig_id, length
        )
        logger.info("Adding contig line to vcf: {0}".format(contig_line))
        self.parse_meta_data(contig_line)
        return