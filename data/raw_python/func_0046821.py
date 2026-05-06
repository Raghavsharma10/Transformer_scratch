def can_sequence_objectives(self):
        """Tests if this user can sequence objectives.

        A return of true does not guarantee successful authorization. A
        return of false indicates that it is known performing any update
        will result in a PermissionDenied. This is intended as a hint to
        an application that may opt not to offer these operations to an
        unauthorized user.

        return: (boolean) - false if sequencing objectives is not
                authorized, true otherwise
        compliance: mandatory - This method must be implemented.

        """
        url_path = construct_url('authorization',
                                 bank_id=self._catalog_idstr)
        return self._get_request(url_path)['objectiveHierarchyHints']['canModifyHierarchy']