def get_relationship_form_for_create(self, source_id=None, destination_id=None, relationship_record_types=None):
        """Gets the relationship form for creating new relationships.

        A new form should be requested for each create transaction.

        arg:    source_id (osid.id.Id): ``Id`` of a peer
        arg:    destination_id (osid.id.Id): ``Id`` of the related peer
        arg:    relationship_record_types (osid.type.Type[]): array of
                relationship record types
        return: (osid.relationship.RelationshipForm) - the relationship
                form
        raise:  NotFound - ``source_id`` or ``destination_id`` is not
                found
        raise:  NullArgument - ``source_id`` or ``destination_id`` is
                ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        raise:  Unsupported - unable to get form for requested recod
                types
        *compliance: mandatory -- This method must be implemented.*

        """
        if source_id is None or destination_id is None:
            raise NullArgument()
        if relationship_record_types is None:
            pass  # Still need to deal with the record_types argument
        relationship_form = objects.RelationshipForm(osid_object_map=None,
                                                     source_id=source_id,
                                                     destination_id=destination_id)
        self._forms[relationship_form.get_id().get_identifier()] = not CREATED
        return relationship_form