def get_objective_nodes(self,
                            objective_id=None,
                            ancestor_levels=None,
                            descendant_levels=None,
                            include_siblings=None):
        """Gets a portion of the hierarchy for the given objective.

        arg:    objective_id (osid.id.Id): the Id to query
        arg:    ancestor_levels (cardinal): the maximum number of
                ancestor levels to include. A value of 0 returns no
                parents in the node.
        arg:    descendant_levels (cardinal): the maximum number of
                descendant levels to include. A value of 0 returns no
                children in the node.
        arg:    include_siblings (boolean): true to include the siblings
                of the given node, false to omit the siblings
        return: (osid.learning.ObjectiveNode) - an objective node
        raise:  NotFound - objective_id not found
        raise:  NullArgument - objective_id is null
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        compliance: mandatory - This method must be implemented.

        """
        if objective_id is None:
            url_path = construct_url('roots',
                                     bank_id=self._catalog_idstr,
                                     descendents=descendant_levels)
            return self._get_request(url_path)
        else:
            raise Unimplemented()