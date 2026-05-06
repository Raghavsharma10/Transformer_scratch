def elements(self) -> devicetools.Elements:
        """A |set| containing the |Node| objects of all handled
        |Selection| objects.

        >>> from hydpy import Selection, Selections
        >>> selections = Selections(
        ...     Selection('sel1', ['node1'], ['element1']),
        ...     Selection('sel2', ['node1'], ['element2', 'element3']))
        >>> selections.elements
        Elements("element1", "element2", "element3")
        """
        elements = devicetools.Elements()
        for selection in self:
            elements += selection.elements
        return elements