def get_relationships_for_destination_on_date(self, destination_id, from_, to):
        """Gets a ``RelationshipList`` corresponding to the given peer ``Id`` with a starting effective date in the given range inclusive.

        arg:    destination_id (osid.id.Id): a peer ``Id``
        arg:    from (osid.calendaring.DateTime): starting date
        arg:    to (osid.calendaring.DateTime): ending date
        return: (osid.relationship.RelationshipList) - the relationships
        raise:  InvalidArgument - ``from is greater than to``
        raise:  NullArgument - ``destination_id, from`` ,or ``to`` is
                ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.relationship.RelationshipLookupSession.get_relationships_for_destination_on_date
        relationship_list = []
        for relationship in self.get_relationships_for_destination():
            if overlap(from_, to, relationship.start_date, relationship.end_date):
                relationship_list.append(relationship)
        return objects.RelationshipList(relationship_list, runtime=self._runtime)