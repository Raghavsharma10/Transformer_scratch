def can_assign_requisites(self):
        """Tests if this user can manage objective requisites.

        A return of true does not guarantee successful authorization. A
        return of false indicates that it is known mapping methods in
        this session will result in a PermissionDenied. This is intended
        as a hint to an application that may opt not to offer assignment
        operations to unauthorized users.

        return: (boolean) - false if mapping is not authorized, true
                otherwise
        compliance: mandatory - This method must be implemented.

        """
        url_path = construct_url('authorization',
                                 bank_id=self._catalog_idstr)
        return self._get_request(url_path)['objectiveRequisiteHints']['canAssign']