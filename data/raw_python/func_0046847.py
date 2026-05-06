def get_objective_banks_by_genus_type(self, objective_bank_genus_type=None):
        """Gets a ObjectiveBankList corresponding to the given objective
        bank genus Type which does not include objective banks of types
        derived from the specified Type.
        In plenary mode, the returned list contains all known objective
        banks or an error results. Otherwise, the returned list may
        contain only those objective banks that are accessible through
        this session.
        arg:    objectiveBankGenusType (osid.type.Type): an objective
                bank genus type
        return: (osid.learning.ObjectiveBankList) - the returned
                ObjectiveBank list
        raise:  NullArgument - objectiveBankGenusType is null
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        compliance: mandatory - This method must be implemented.

        """
        if objective_bank_genus_type is None:
            raise NullArgument()
        url_path = construct_url('objective_banks_by_genus',
                                 genus_type=objective_bank_genus_type)
        return objects.ObjectiveBankList(self._get_request(url_path))