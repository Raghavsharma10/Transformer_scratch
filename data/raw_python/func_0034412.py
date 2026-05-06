def get_breakpoint_graph(stream, merge_edges=True):
        """ Taking a file-like object transforms supplied gene order data into the language of

        :param merge_edges: a flag that indicates if parallel edges in produced breakpoint graph shall be merged or not
        :type merge_edges: ``bool``
        :param stream: any iterable object where each iteration produces a ``str`` object
        :type stream: ``iterable`` ver ``str``
        :return: an instance of a BreakpointGraph that contains information about adjacencies in genome specified in GRIMM formatted input
        :rtype: :class:`bg.breakpoint_graph.BreakpointGraph`
        """
        result = BreakpointGraph()
        current_genome = None
        fragment_data = {}
        for line in stream:
            line = line.strip()
            if len(line) == 0:
                ###############################################################################################
                #
                # empty lines are omitted
                #
                ###############################################################################################
                continue
            if GRIMMReader.is_genome_declaration_string(data_string=line):
                ###############################################################################################
                #
                # is we have a genome declaration, we must update current genome
                # all following gene order data (before EOF or next genome declaration) will be attributed to current genome
                #
                ###############################################################################################
                current_genome = GRIMMReader.parse_genome_declaration_string(data_string=line)
                fragment_data = {}
            elif GRIMMReader.is_comment_string(data_string=line):
                if GRIMMReader.is_comment_data_string(string=line):
                    path, (key, value) = GRIMMReader.parse_comment_data_string(comment_data_string=line)
                    if len(path) > 0 and path[0] == "fragment":
                        add_to_dict_with_path(destination_dict=fragment_data, key=key, value=value, path=path)
                else:
                    continue
            elif current_genome is not None:
                ###############################################################################################
                #
                # gene order information that is specified before the first genome is specified can not be attributed to anything
                # and thus omitted
                #
                ###############################################################################################
                parsed_data = GRIMMReader.parse_data_string(data_string=line)
                edges = GRIMMReader.get_edges_from_parsed_data(parsed_data=parsed_data)
                for v1, v2 in edges:
                    edge_specific_data = {
                        "fragment": {
                            "forward_orientation": (v1, v2)
                        }
                    }
                    edge = BGEdge(vertex1=v1, vertex2=v2, multicolor=Multicolor(current_genome), data=deepcopy(fragment_data))
                    edge.update_data(source=edge_specific_data)
                    result.add_bgedge(bgedge=edge,
                                      merge=merge_edges)
        return result