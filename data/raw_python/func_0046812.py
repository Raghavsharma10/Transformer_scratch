def get_root_objectives(self):
        """Gets the root objective in this objective hierarchy.

        return: (osid.learning.ObjectiveList) - the root objective
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        compliance: mandatory - This method is must be implemented.

        """
        url_path = construct_url('roots',
                                 bank_id=self._catalog_idstr)
        return objects.ObjectiveList(self._get_request(url_path))