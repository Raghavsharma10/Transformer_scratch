def get_parent_bank_nodes(self):
        """Gets the parents of this bank.

        return: (osid.assessment.BankNodeList) - the parents of this
                node
        *compliance: mandatory -- This method must be implemented.*

        """
        parent_bank_nodes = []
        for node in self._my_map['parentNodes']:
            parent_bank_nodes.append(BankNode(
                node._my_map,
                runtime=self._runtime,
                proxy=self._proxy,
                lookup_session=self._lookup_session))
        return BankNodeList(parent_bank_nodes)