def get_relationships_by_genus_type(self, relationship_genus_type=None):
        """Gets a ``RelationshipList`` corresponding to the given relationship genus ``Type``
            which does not include relationships of types derived from the specified ``Type``.

        arg:    relationship_genus_type (osid.type.Type): a relationship
                genus type
        return: (osid.relationship.RelationshipList) - the returned
                ``Relationship list``
        raise:  NullArgument - ``relationship_genus_type`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        if relationship_genus_type is None:
            raise NullArgument()
        url_path = ('/handcar/services/relationship/families/' +
                    self._catalog_idstr + '/relationships?genustypeid=' +
                    relationship_genus_type.get_identifier())
        return objects.RelationshipList(self._get_request(url_path))