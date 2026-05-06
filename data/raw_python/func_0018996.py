def variables(self) -> Set[str]:
        """A set of all different |Node.variable| values of the |Node|
        objects directly connected to the actual |Element| object.

        Suppose there is an element connected to five nodes, which (partly)
        represent different variables:

        >>> from hydpy import Element, Node
        >>> element = Element('Test',
        ...                   inlets=(Node('N1', 'X'), Node('N2', 'Y1')),
        ...                   outlets=(Node('N3', 'X'), Node('N4', 'Y2')),
        ...                   receivers=(Node('N5', 'X'), Node('N6', 'Y3')),
        ...                   senders=(Node('N7', 'X'), Node('N8', 'Y4')))

        Property |Element.variables| puts all the different variables of
        these nodes together:

        >>> sorted(element.variables)
        ['X', 'Y1', 'Y2', 'Y3', 'Y4']
        """
        variables: Set[str] = set()
        for connection in self.__connections:
            variables.update(connection.variables)
        return variables