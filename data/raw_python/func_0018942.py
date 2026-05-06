def select_nodenames(self, *substrings: str) -> 'Selection':
        """Restrict the current selection to all nodes with a name
        containing at least one of the given substrings  (does not
        affect any elements).

        See the documentation on method |Selection.search_nodenames| for
        additional information.
        """
        self.nodes = self.search_nodenames(*substrings).nodes
        return self