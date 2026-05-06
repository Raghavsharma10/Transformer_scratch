def get_objective_bank_hierarchy(self, alias):
        """Gets the hierarchy associated with this session.

        return: (osid.hierarchy.Hierarchy) - the hierarchy associated
                with this session
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        url_path = self._urls.hierarchy(re.sub(r'[ ]', '', alias.lower()))
        return self._get_request(url_path)