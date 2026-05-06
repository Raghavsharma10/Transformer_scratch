def get_activities(self):
        """Gets all Activities.
        In plenary mode, the returned list contains all known activites
        or an error results. Otherwise, the returned list may contain
        only those activities that are accessible through this session.
        return: (osid.learning.ActivityList) - a ActivityList
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        compliance: mandatory - This method must be implemented.

        """
        url_path = construct_url('activities',
                                 bank_id=self._catalog_idstr)
        return objects.ActivityList(self._get_request(url_path))