def get_relationship(self, relationship_id=None):
        """Gets the ``Relationship`` specified by its ``Id``.

        arg:    relationship_id (osid.id.Id): the ``Id`` of the
                ``Relationship`` to retrieve
        return: (osid.relationship.Relationship) - the returned
                ``Relationship``
        raise:  NotFound - no ``Relationship`` found with the given
                ``Id``
        raise:  NullArgument - ``relationship_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        if relationship_id is None:
            raise NullArgument()
        url_path = ('/handcar/services/relationship/families/' +
                    self._catalog_idstr +
                    '/relationships/' + str(relationship_id))
        return objects.Relationship(self._get_request(url_path))