def delete_relationship(self, relationship_id=None):
        """Deletes a ``Relationship``.

        arg:    relationship_id (osid.id.Id): the ``Id`` of the
                ``Relationship`` to remove
        raise:  NotFound - ``relationship_id`` not found
        raise:  NullArgument - ``relationship_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        if relationship_id is None:
            raise NullArgument()
        if not isinstance(relationship_id, Id):
            raise InvalidArgument('argument type is not an osid Id')

        url_path = ('/handcar/services/relationship/families/' +
                    self._catalog_idstr + '/relationships/' +
                    str(relationship_id))
        result = self._delete_request(url_path)
        return objects.Relationship(result)