def selection(self):
        """A complete |Selection| object of all "supplying" and "routing"
        elements and required nodes.

        >>> from hydpy import RiverBasinNumbers2Selection
        >>> rbns2s = RiverBasinNumbers2Selection(
        ...                            (111, 113, 1129, 11269, 1125, 11261,
        ...                             11262, 1123, 1124, 1122, 1121))
        >>> rbns2s.selection
        Selection("complete",
                  nodes=("node_1123", "node_1125", "node_11269", "node_1129",
                         "node_113", "node_outlet"),
                  elements=("land_111", "land_1121", "land_1122", "land_1123",
                            "land_1124", "land_1125", "land_11261",
                            "land_11262", "land_11269", "land_1129",
                            "land_113", "stream_1123", "stream_1125",
                            "stream_11269", "stream_1129", "stream_113"))

        Besides the possible modifications on the names of the different
        nodes and elements, the name of the selection can be set differently:

        >>> rbns2s.selection_name = 'sel'
        >>> from hydpy import pub
        >>> with pub.options.ellipsis(1):
        ...     print(repr(rbns2s.selection))
        Selection("sel",
                  nodes=("node_1123", ...,"node_outlet"),
                  elements=("land_111", ...,"stream_113"))
        """
        return selectiontools.Selection(
            self.selection_name, self.nodes, self.elements)