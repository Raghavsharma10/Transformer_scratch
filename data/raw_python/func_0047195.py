def get_proficiencies_by_genus_type_for_resource_on_date(self, resource_id, proficiency_genus_type, from_, to):
        """Gets a ``ProficiencyList`` of the given proficiency genus type relating to the given resource effective during the entire given date range inclusive but not confined to the date range.

        arg:    resource_id (osid.id.Id): a resource ``Id``
        arg:    proficiency_genus_type (osid.type.Type): a proficiency
                genus type
        arg:    from (osid.calendaring.DateTime): starting date
        arg:    to (osid.calendaring.DateTime): ending date
        return: (osid.learning.ProficiencyList) - the returned
                ``Proficiency`` list
        raise:  InvalidArgument - ``from`` is greater than ``to``
        raise:  NullArgument - ``resource_id, proficiency_genus_type,
                from`` or ``to`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.relationship.RelationshipLookupSession.get_relationships_by_genus_type_for_source_on_date
        proficiency_list = []
        for proficiency in self.get_proficiencies_by_genus_type_for_resource():
            if overlap(from_, to, proficiency.start_date, proficiency.end_date):
                proficiency_list.append(proficiency)
        return objects.ProficiencyList(proficiency_list, runtime=self._runtime)