def get_relationships_by_ids(self, relationship_ids=None):
        """Gets a ``RelationshipList`` corresponding to the given ``IdList``.

        arg:    relationship_ids (osid.id.IdList): the list of ``Ids``
                to retrieve
        return: (osid.relationship.RelationshipList) - the returned
                ``Relationship list``
        raise:  NotFound - an ``Id`` was not found
        raise:  NullArgument - ``relationship_ids`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        if relationship_ids is None:
            raise NullArgument()
        relationships = []
        for i in relationship_ids:
            relationship = None
            url_path = ('/handcar/services/relationship/families/' +
                        self._catalog_idstr +
                        '/relatioships/' + str(i))
            try:
                relationship = self._get_request(url_path)
            except (NotFound, OperationFailed):
                if self._relationship_view == PLENARY:
                    raise
                else:
                    pass
            if relationship:
                if not (self._relationship_view == COMPARATIVE and
                        relationship in relationships):
                    relationships.append(relationship)
        return objects.RelationshipList(relationships)