def get_family(self):
        """Gets the ``Family`` associated with this session.

        return: (osid.relationship.Family) - the family
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        return FamilyLookupSession(proxy=self._proxy,
                                   runtime=self._runtime).get_family(self._family_id)