def merge(cls, edge1, edge2):
        """ Merges multi-color information from two supplied :class:`BGEdge` instances into a new :class:`BGEdge`

        Since :class:`BGEdge` represents an undirected edge, created edge's vertices are assign according to the order in first supplied edge.

        Accounts for subclassing.

        :param edge1: first out of two edge information from which is to be merged into a new one
        :param edge2: second out of two edge information from which is to be merged into a new one
        :return: a new undirected with multi-color information merged from two supplied :class:`BGEdge` objects
        :raises: ``ValueError``
        """
        if edge1.vertex1 != edge2.vertex1 and edge1.vertex1 != edge2.vertex2:
            raise ValueError("Edges to be merged do not connect same vertices")
        forward = edge1.vertex1 == edge2.vertex1
        if forward and edge1.vertex2 != edge2.vertex2:
            raise ValueError("Edges to be merged do not connect same vertices")
        elif not forward and edge1.vertex2 != edge2.vertex1:
            raise ValueError("Edges to be merged do not connect same vertices")
        return cls(vertex1=edge1.vertex1, vertex2=edge1.vertex2, multicolor=edge1.multicolor + edge2.multicolor)