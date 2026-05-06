def get_edges_from_parsed_data(parsed_data):
        """ Taking into account fragment type (circular|linear) and retrieved gene order information translates adjacencies between blocks into edges for addition to the :class:`bg.breakpoint_graph.BreakpointGraph`

        In case supplied fragment is linear (``$``) special artificial vertices (with ``__infinity`` suffix) are introduced to denote fragment extremities

        :param parsed_data: (``$`` | ``@``, [(``+`` | ``-``, block_name),...]) formatted data about fragment type and ordered list of oriented blocks
        :type parsed_data: ``tuple(str, list((str, str), ...))``
        :return: a list of vertices pairs that would correspond to edges in :class:`bg.breakpoint_graph.BreakpointGraph`
        :rtype: ``list((str, str), ...)``
        """
        chr_type, blocks = parsed_data
        vertices = []
        for block in blocks:
            ###############################################################################################
            #
            # each block is represented as a pair of vertices (that correspond to block extremities)
            #
            ###############################################################################################
            v1, v2 = GRIMMReader.__assign_vertex_pair(block)
            vertices.append(v1)
            vertices.append(v2)
        if chr_type == "@":
            ###############################################################################################
            #
            # if we parse a circular genomic fragment we must introduce an additional pair of vertices (edge)
            # that would connect two outer most vertices in the vertex list, thus connecting fragment extremities
            #
            ###############################################################################################
            vertex = vertices.pop()
            vertices.insert(0, vertex)
        elif chr_type == "$":
            ###############################################################################################
            #
            # if we parse linear genomic fragment, we introduce two artificial (infinity) vertices
            # that correspond to fragments ends, and introduce edges between them and respective outermost block vertices
            #
            # if outermost vertices at this moment are repeat vertices, the outermost pair shall be discarded and the innermost
            # vertex info shall be utilized in the infinity vertex, that is introduced for the fragment extremity
            #
            ###############################################################################################
            if vertices[0].is_repeat_vertex:
                left_iv_tags = sorted([(tag, value) if tag != "repeat" else (tag, BGVertex.get_vertex_name_root(vertices[1].name))
                                       for tag, value in vertices[1].tags])
                left_iv_root_name = BGVertex.get_vertex_name_root(vertices[2].name)
                vertices = vertices[2:]
            else:
                left_iv_tags = []
                left_iv_root_name = vertices[0].name
            if vertices[-1].is_repeat_vertex:
                right_iv_tags = sorted(
                        [(tag, value) if tag != "repeat" else (tag, BGVertex.get_vertex_name_root(vertices[-2].name))
                         for tag, value in vertices[-2].tags])
                right_iv_root_name = BGVertex.get_vertex_name_root(vertices[-3].name)
                vertices = vertices[:-2]
            else:
                right_iv_tags = []
                right_iv_root_name = BGVertex.get_vertex_name_root(vertices[-1].name)
            left_iv, right_iv = TaggedInfinityVertex(left_iv_root_name), TaggedInfinityVertex(right_iv_root_name)
            left_iv.tags = left_iv_tags
            right_iv.tags = right_iv_tags
            vertices.insert(0, left_iv)
            vertices.append(right_iv)
        return [(v1, v2) for v1, v2 in zip(vertices[::2], vertices[1::2])]