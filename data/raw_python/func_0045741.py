def get_family(self, family_id=None):
        """Gets the ``Family`` specified by its ``Id``.

        In plenary mode, the exact ``Id`` is found or a ``NotFound``
        results. Otherwise, the returned ``Family`` may have a different
        ``Id`` than requested, such as the case where a duplicate ``Id``
        was assigned to a ``Family`` and retained for compatibil

        arg:    family_id (osid.id.Id): ``Id`` of the ``Family``
        return: (osid.relationship.Family) - the family
        raise:  NotFound - ``family_id`` not found
        raise:  NullArgument - ``family_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method is must be implemented.*

        """
        if family_id is None:
            raise NullArgument()
        url_path = '/handcar/services/relationship/families/' + str(family_id)
        return objects.Family(self._get_request(url_path))