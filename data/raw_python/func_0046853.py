def get_objective_bank_record_types(self):
        """Gets the objective bank types available in Handcar.
        arg:    None
        return: (osid.type.TypeList) - list of objective bank types
        raise:  NotFound - objectiveBankTypes is not found
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        compliance: mandatory - This method must be implemented.

        """
        try:
            url_path = construct_url('objective_bank_types')
            objective_bank_types = typeObjects.TypeList(self._get_request(url_path))
        except Exception:
            raise
        return objective_bank_types