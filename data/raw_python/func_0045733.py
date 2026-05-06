def get_relationships_by_genus_type_for_destination(self, destination_id=None, relationship_genus_type=None):
        """Gets a ``RelationshipList`` corresponding to the given peer ``Id`` and relationship genus ``Type.

        Relationships`` of any genus derived from the given genus are
        returned.

        In plenary mode, the returned list contains all of the
        relationships corresponding to the given peer, including
        duplicates, or an error results if a relationship is
        inaccessible. Otherwise, inaccessible ``Relationships`` may be
        omitted from the list and may present the elements in any order
        including returning a unique set.

        In effective mode, relationships are returned that are currently
        effective. In any effective mode, effective relationships and
        those currently expired are returned.

        arg:    destination_id (osid.id.Id): a peer ``Id``
        arg:    relationship_genus_type (osid.type.Type): a relationship
                genus type
        return: (osid.relationship.RelationshipList) - the relationships
        raise:  NullArgument - ``destination_id`` or
                ``relationship_genus_type`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        if destination_id is None or relationship_genus_type is None:
            raise NullArgument()
        url_path = ('/handcar/services/relationship/families/' +
                    self._catalog_idstr + '/relationships?genustypeid=' +
                    relationship_genus_type.get_identifier + '?sourceid=' +
                    str(destination_id))
        return objects.RelationshipList(self._get_request(url_path))