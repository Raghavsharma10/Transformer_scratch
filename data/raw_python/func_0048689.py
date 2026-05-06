def get_parent_bin_nodes(self):
        """Gets the parents of this bin.

        return: (osid.resource.BinNodeList) - the parents of the ``id``
        *compliance: mandatory -- This method must be implemented.*

        """
        parent_bin_nodes = []
        for node in self._my_map['parentNodes']:
            parent_bin_nodes.append(BinNode(
                node._my_map,
                runtime=self._runtime,
                proxy=self._proxy,
                lookup_session=self._lookup_session))
        return BinNodeList(parent_bin_nodes)