def can_update_activities(self):
        """Tests if this user can update Activities.
        A return of true does not guarantee successful authorization. A
        return of false indicates that it is known updating an Activity
        will result in a PermissionDenied. This is intended as a hint
        to an application that may opt not to offer update operations to
        an unauthorized user.
        return: (boolean) - false if activity modification is not
                authorized, true otherwise
        compliance: mandatory - This method must be implemented.

        """
        url_path = construct_url('authorization',
                                 bank_id=self._catalog_idstr)
        return self._get_request(url_path)['activityHints']['canUpdate']