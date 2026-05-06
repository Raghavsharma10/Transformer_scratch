def get_root_objective_ids(self):
        """Gets the root objective Ids in this hierarchy.

        return: (osid.id.IdList) - the root objective Ids
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        compliance: mandatory - This method must be implemented.

        """
        url_path = construct_url('rootids',
                                 bank_id=self._catalog_idstr)
        id_list = list()
        for identifier in self._get_request(url_path)['ids']:
            id_list.append(Id(idstr=identifier))
        return id_objects.IdList(id_list)