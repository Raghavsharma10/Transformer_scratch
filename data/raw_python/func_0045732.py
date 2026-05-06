def get_relationships_for_destination(self, destination_id=None):
        """Gets a ``RelationshipList`` corresponding to the given peer ``Id``.

        arg:    destination_id (osid.id.Id): a peer ``Id``
        return: (osid.relationship.RelationshipList) - the relationships
        raise:  NullArgument - ``destination_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        if destination_id is None:
            raise NullArgument()
        url_path = ('/handcar/services/relationship/families/' +
                    self._catalog_idstr + '/relationships?sourceid=' +
                    str(destination_id))
        return objects.RelationshipList(self._get_request(url_path))