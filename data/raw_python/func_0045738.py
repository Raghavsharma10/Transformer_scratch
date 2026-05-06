def get_relationship_form_for_update(self, relationship_id=None):
        """Gets the relationship form for updating an existing relationship.

        A new relationship form should be requested for each update
        transaction.

        arg:    relationship_id (osid.id.Id): the ``Id`` of the
                ``Relationship``
        return: (osid.relationship.RelationshipForm) - the relationship
                form
        raise:  NotFound - ``relationship_id`` is not found
        raise:  NullArgument - ``relationship_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        if relationship_id is None:
            raise NullArgument()
        try:
            url_path = ('/handcar/services/relationship/families/' +
                        self._catalog_idstr + '/relationships/' + str(relationship_id))
            relationship = objects.Relationship(self._get_request(url_path))
        except Exception:
            raise
        relationship_form = objects.RelationshipForm(relationship._my_map)
        self._forms[relationship_form.get_id().get_identifier()] = not UPDATED
        return relationship_form