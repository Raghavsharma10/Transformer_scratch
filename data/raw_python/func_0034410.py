def __assign_vertex_pair(block):
        """ Assigns usual BreakpointGraph type vertices to supplied block.

        Vertices are labeled as "block_name" + "h" and "block_name" + "t" according to blocks orientation.

        :param block: information about a genomic block to create a pair of vertices for in a format of ( ``+`` | ``-``, block_name)
        :type block: ``(str, str)``
        :return: a pair of vertices labeled according to supplied blocks name (respecting blocks orientation)
        :rtype: ``(str, str)``
        """
        sign, name = block
        data = name.split(BlockVertex.NAME_SEPARATOR)
        root_name, data = data[0], data[1:]
        tags = [entry.split(TaggedVertex.TAG_SEPARATOR) for entry in data]
        for tag_entry in tags:
            if len(tag_entry) == 1:
                tag_entry.append(None)
            elif len(tag_entry) > 2:
                tag_entry[1:] = [TaggedVertex.TAG_SEPARATOR.join(tag_entry[1:])]
        tail, head = root_name + "t", root_name + "h"
        tail, head = TaggedBlockVertex(tail), TaggedBlockVertex(head)
        tail.mate_vertex = head
        head.mate_vertex = tail
        for tag, value in tags:
            head.add_tag(tag, value)
            tail.add_tag(tag, value)
        return (tail, head) if sign == "+" else (head, tail)