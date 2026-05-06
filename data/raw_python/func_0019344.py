def supplier_elements(self):
        """A |Elements| collection of all "supplying" basins.

        (All river basins are assumed to supply something to the downstream
        basin.)

        >>> from hydpy import RiverBasinNumbers2Selection
        >>> rbns2s = RiverBasinNumbers2Selection(
        ...                            (111, 113, 1129, 11269, 1125, 11261,
        ...                             11262, 1123, 1124, 1122, 1121))

        The following elements are properly connected to the required
        outlet nodes already:

        >>> for element in rbns2s.supplier_elements:
        ...     print(repr(element))
        Element("land_111",
                outlets="node_113")
        Element("land_1121",
                outlets="node_1123")
        Element("land_1122",
                outlets="node_1123")
        Element("land_1123",
                outlets="node_1125")
        Element("land_1124",
                outlets="node_1125")
        Element("land_1125",
                outlets="node_1129")
        Element("land_11261",
                outlets="node_11269")
        Element("land_11262",
                outlets="node_11269")
        Element("land_11269",
                outlets="node_1129")
        Element("land_1129",
                outlets="node_113")
        Element("land_113",
                outlets="node_outlet")

        It is both possible to change the prefix names of the elements
        and nodes, as long as it results in a valid variable name (e.g.
        does not start with a number):

        >>> rbns2s.supplier_prefix = 'a_'
        >>> rbns2s.node_prefix = 'b_'
        >>> rbns2s.supplier_elements
        Elements("a_111", "a_1121", "a_1122", "a_1123", "a_1124", "a_1125",
                 "a_11261", "a_11262", "a_11269", "a_1129", "a_113")
        """
        elements = devicetools.Elements()
        for supplier in self._supplier_numbers:
            element = self._get_suppliername(supplier)
            try:
                outlet = self._get_nodename(self._up2down[supplier])
            except TypeError:
                outlet = self.last_node
            elements += devicetools.Element(element, outlets=outlet)
        return elements