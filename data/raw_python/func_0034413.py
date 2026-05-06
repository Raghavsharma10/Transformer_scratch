def get_blocks_in_grimm_from_breakpoint_graph(bg):
        """
        :param bg: a breakpoint graph, that contians all the information
        :type bg: ``bg.breakpoint_graph.BreakpointGraph``
        :return: list of strings, which represent genomes present in breakpoint graph as orders of blocks and is compatible with GRIMM format
        """
        result = []
        genomes = bg.get_overall_set_of_colors()
        for genome in genomes:
            genome_graph = bg.get_genome_graph(color=genome)
            genome_blocks_orders = genome_graph.get_blocks_order()
            blocks_orders = genome_blocks_orders[genome]
            if len(blocks_orders) > 0:
                result.append(">{genome_name}".format(genome_name=genome.name))
            for chr_type, blocks_order in blocks_orders:
                string = " ".join(value if sign == "+" else sign + value for sign, value in blocks_order)
                string += " {chr_type}".format(chr_type=chr_type)
                result.append(string)
        return result