def get_parent_objective_bank_nodes(self):
        """Gets the parents of this objective bank.

        return: (osid.learning.ObjectiveBankNodeList) - the parents of
                the ``id``
        *compliance: mandatory -- This method must be implemented.*

        """
        parent_objective_bank_nodes = []
        for node in self._my_map['parentNodes']:
            parent_objective_bank_nodes.append(ObjectiveBankNode(
                node._my_map,
                runtime=self._runtime,
                proxy=self._proxy,
                lookup_session=self._lookup_session))
        return ObjectiveBankNodeList(parent_objective_bank_nodes)