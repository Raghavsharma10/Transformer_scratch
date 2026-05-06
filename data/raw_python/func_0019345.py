def router_elements(self):
        """A |Elements| collection of all "routing" basins.

        (Only river basins with a upstream basin are assumed to route
        something to the downstream basin.)

        >>> from hydpy import RiverBasinNumbers2Selection
        >>> rbns2s = RiverBasinNumbers2Selection(
        ...                            (111, 113, 1129, 11269, 1125, 11261,
        ...                             11262, 1123, 1124, 1122, 1121))

        The following elements are properly connected to the required
        inlet and outlet nodes already:

        >>> for element in rbns2s.router_elements:
        ...     print(repr(element))
        Element("stream_1123",
                inlets="node_1123",
                outlets="node_1125")
        Element("stream_1125",
                inlets="node_1125",
                outlets="node_1129")
        Element("stream_11269",
                inlets="node_11269",
                outlets="node_1129")
        Element("stream_1129",
                inlets="node_1129",
                outlets="node_113")
        Element("stream_113",
                inlets="node_113",
                outlets="node_outlet")

        It is both possible to change the prefix names of the elements
        and nodes, as long as it results in a valid variable name (e.g.
        does not start with a number):

        >>> rbns2s.router_prefix = 'c_'
        >>> rbns2s.node_prefix = 'd_'
        >>> rbns2s.router_elements
        Elements("c_1123", "c_1125", "c_11269", "c_1129", "c_113")
        """
        elements = devicetools.Elements()
        for router in self._router_numbers:
            element = self._get_routername(router)
            inlet = self._get_nodename(router)
            try:
                outlet = self._get_nodename(self._up2down[router])
            except TypeError:
                outlet = self.last_node
            elements += devicetools.Element(
                element, inlets=inlet, outlets=outlet)
        return elements