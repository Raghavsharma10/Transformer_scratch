def get_relationships_by_genus_type_for_source_on_date(self, source_id, relationship_genus_type, from_, to):
        """Gets a ``RelationshipList`` corresponding to the given peer ``Id`` and relationship genus ``Type`` and effective during the entire given date range inclusive but not confined to the date range.

        arg:    source_id (osid.id.Id): a peer ``Id``
        arg:    relationship_genus_type (osid.type.Type): a relationship
                genus type
        arg:    from (osid.calendaring.DateTime): starting date
        arg:    to (osid.calendaring.DateTime): ending date
        return: (osid.relationship.RelationshipList) - the relationships
        raise:  InvalidArgument - ``from is greater than to``
        raise:  NullArgument - ``source_id, relationship_genus_type,
                from`` or ``to`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.relationship.RelationshipLookupSession.get_relationships_by_genus_type_for_source_on_date
        relationship_list = []
        for relationship in self.get_relationships_by_genus_type_for_source():
            if overlap(from_, to, relationship.start_date, relationship.end_date):
                relationship_list.append(relationship)
        return objects.RelationshipList(relationship_list, runtime=self._runtime)