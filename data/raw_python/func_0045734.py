def get_relationships(self):
        """Gets all ``Relationships``.

        return: (osid.relationship.RelationshipList) - a list of
                ``Relationships``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        url_path = ('/handcar/services/relationship/families/' +
                    self._catalog_idstr + '/relationships')
        return objects.RelationshipList(self._get_request(url_path))