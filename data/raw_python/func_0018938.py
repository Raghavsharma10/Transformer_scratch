def select_upstream(self, device: devicetools.Device) -> 'Selection':
        """Restrict the current selection to the network upstream of the given
        starting point, including the starting point itself.

        See the documentation on method |Selection.search_upstream| for
        additional information.
        """
        upstream = self.search_upstream(device)
        self.nodes = upstream.nodes
        self.elements = upstream.elements
        return self