def can_access_objective_hierarchy(self):
        """Tests if this user can perform hierarchy queries.

        A return of true does not guarantee successful authorization. A
        return of false indicates that it is known all methods in this
        session will result in a PermissionDenied. This is intended as a
        hint to an an application that may not offer traversal functions
        to unauthorized users.

        return: (boolean) - false if hierarchy traversal methods are not
                authorized, true otherwise
        compliance: mandatory - This method must be implemented.

        """
        url_path = construct_url('authorization',
                                 bank_id=self._catalog_idstr)
        return self._get_request(url_path)['objectiveHierarchyHints']['canAccessHierarchy']