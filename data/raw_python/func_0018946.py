def deselect_elementnames(self, *substrings: str) -> 'Selection':
        """Restrict the current selection to all elements with a name
        not containing at least one of the given substrings.   (does
        not affect any nodes).

        See the documentation on method |Selection.search_elementnames| for
        additional information.
        """
        self.elements -= self.search_elementnames(*substrings).elements
        return self