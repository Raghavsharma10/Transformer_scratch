def get_objective_bank_nodes(self,
                                 objective_bank_id=None,
                                 ancestor_levels=None,
                                 descendant_levels=None,
                                 include_siblings=None):
        """Gets a portion of the hierarchy for the given objective bank.

        arg:    includeSiblings (boolean): true to include the siblings
                of the given node, false to omit the siblings
        return: (osid.learning.ObjectiveBankNode) - an objective bank
                node
        raise:  NotFound - objectiveBankId not found
        raise:  NullArgument - objectiveBankId is null
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        compliance: mandatory - This method must be implemented.

        """
        if descendant_levels:
            url_path = self._urls.nodes(alias=objective_bank_id, depth=descendant_levels)
        else:
            url_path = self._urls.nodes(alias=objective_bank_id)
        return self._get_request(url_path)