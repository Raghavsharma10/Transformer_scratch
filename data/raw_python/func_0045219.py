def add_format(self, format_id, number, entry_type, description):
        """
        Add a format line to the header.

        Arguments:
            format_id (str): The id of the format line
            number (str): Integer or any of [A,R,G,.]
            entry_type (str): Any of [Integer,Float,Flag,Character,String]
            description (str): A description of the info line

        """
        format_line = '##FORMAT=<ID={0},Number={1},Type={2},Description="{3}">'.format(
            format_id, number, entry_type, description
        )
        logger.info("Adding format line to vcf: {0}".format(format_line))
        self.parse_meta_data(format_line)
        return