def format(self, node_formatter=None, separator='/'):
        # type: (Optional[Callable[[PathParam], str]]) -> str
        """
        Format a URL path.
        
        An optional `node_parser(PathNode)` can be supplied for converting a 
        `PathNode` into a string to support the current web framework.  
        
        """
        if self._nodes == ('',):
            return separator
        else:
            node_formatter = node_formatter or self.odinweb_node_formatter
            return separator.join(node_formatter(n) if isinstance(n, PathParam) else n for n in self._nodes)