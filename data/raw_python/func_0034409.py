def parse_data_string(data_string):
        """ Parses a string assumed to contain gene order data, retrieving information about fragment type, gene order, blocks names and their orientation

        First checks if gene order termination signs are present.
        Selects the earliest one.
        Checks that information preceding is not empty and contains gene order.
        Generates results structure by retrieving information about fragment type, blocks names and orientations.

        **NOTE:** comment signs do not work in data strings. Rather use the fact that after first gene order termination sign everything is ignored for processing

        :param data_string: a string to retrieve gene order information from
        :type data_string: ``str``
        :return: (``$`` | ``@``, [(``+`` | ``-``, block_name),...]) formatted structure corresponding to gene order in supplied data string and containing fragments type
        :rtype: ``tuple(str, list((str, str), ...))``
        """
        data_string = data_string.strip()
        linear_terminator_index = data_string.index("$") if "$" in data_string else -1
        circular_terminator_index = data_string.index("@") if "@" in data_string else -1
        if linear_terminator_index < 0 and circular_terminator_index < 0:
            raise ValueError("Invalid data string. No chromosome termination sign ($|@) found.")
        if linear_terminator_index == 0 or circular_terminator_index == 0:
            raise ValueError("Invalid data string. No data found before chromosome was terminated.")
        if linear_terminator_index < 0 or 0 < circular_terminator_index < linear_terminator_index:
            ###############################################################################################
            #
            # we either encountered only a circular chromosome termination sign
            # or we have encountered it before we've encountered the circular chromosome termination sign first
            #
            ###############################################################################################
            chr_type = "@"
            terminator_index = circular_terminator_index
        else:
            chr_type = "$"
            terminator_index = linear_terminator_index
        ###############################################################################################
        #
        # everything after first fragment termination sign is omitted
        #
        ###############################################################################################
        data = data_string[:terminator_index].strip()
        ###############################################################################################
        #
        # genomic blocks are separated between each other by the space character
        #
        ###############################################################################################
        split_data = data.split()
        blocks = []
        for block in split_data:
            ###############################################################################################
            #
            # since positively oriented blocks can be denoted both as "+block" as well as "block"
            # we need to figure out where "block" name starts
            #
            ###############################################################################################
            cut_index = 1 if block.startswith("-") or block.startswith("+") else 0
            if cut_index == 1 and len(block) == 1:
                ###############################################################################################
                #
                # block can not be empty
                # from this one can derive the fact, that names "+" and "-" for blocks are forbidden
                #
                ###############################################################################################
                raise ValueError("Empty block name definition")
            blocks.append(("-" if block.startswith("-") else "+", block[cut_index:]))
        return chr_type, blocks