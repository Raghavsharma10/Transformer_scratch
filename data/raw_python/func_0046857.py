def get_root_objective_bank_ids(self, alias):
        """Gets the root objective bank Ids in this hierarchy.

        return: (osid.id.IdList) - the root objective bank Ids
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        compliance: mandatory - This method must be implemented.

        """
        url_path = self._urls.roots(alias)
        return self._get_request(url_path)