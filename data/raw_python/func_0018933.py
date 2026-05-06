def nodes(self) -> devicetools.Nodes:
        """A |set| containing the |Node| objects of all handled
        |Selection| objects.

        >>> from hydpy import Selection, Selections
        >>> selections = Selections(
        ...     Selection('sel1', ['node1', 'node2'], ['element1']),
        ...     Selection('sel2', ['node1', 'node3'], ['element2']))
        >>> selections.nodes
        Nodes("node1", "node2", "node3")
        """
        nodes = devicetools.Nodes()
        for selection in self:
            nodes += selection.nodes
        return nodes