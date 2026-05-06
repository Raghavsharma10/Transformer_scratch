def get_authorizations_on_date(self, from_, to):
        """Gets an ``AuthorizationList`` effective during the entire given date range inclusive but not confined to the date range.

        arg:    from (osid.calendaring.DateTime): starting date
        arg:    to (osid.calendaring.DateTime): ending date
        return: (osid.authorization.AuthorizationList) - the returned
                ``Authorization`` list
        raise:  InvalidArgument - ``from`` is greater than ``to``
        raise:  NullArgument - ``from`` or ``to`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure occurred
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.relationship.RelationshipLookupSession.get_relationships_on_date
        authorization_list = []
        for authorization in self.get_authorizations():
            if overlap(from_, to, authorization.start_date, authorization.end_date):
                authorization_list.append(authorization)
        return objects.AuthorizationList(authorization_list, runtime=self._runtime)