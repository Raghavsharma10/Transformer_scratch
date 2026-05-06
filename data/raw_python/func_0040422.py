def from_file(cls,
                  source,
                  distance_weights=None,
                  merge_same_words=False,
                  group_marker_opening='<<',
                  group_marker_closing='>>'):
        """
        Read a string from a file and derive a ``Graph`` from it.

        This is a convenience function for opening a file and passing its
        contents to ``Graph.from_string()`` (see that for more detail)

        Args:
            source (str): the file to read and derive the graph from
            distance_weights (dict): dict of relative indices corresponding
                with word weights. See ``Graph.from_string`` for more detail.
            merge_same_words (bool): whether nodes which have the same value
                should be merged or not.
            group_marker_opening (str): The string used to mark the beginning
                of word groups.
            group_marker_closing (str): The string used to mark the end
                of word groups.

        Returns: Graph

        Example:
            >>> graph = Graph.from_file('cage.txt')            # doctest: +SKIP
            >>> ' '.join(graph.pick().value for i in range(8)) # doctest: +SKIP
            'poetry i have nothing to say and i'
        """
        source_string = open(source, 'r').read()
        return cls.from_string(source_string,
                               distance_weights,
                               merge_same_words,
                               group_marker_opening=group_marker_opening,
                               group_marker_closing=group_marker_closing)