def get_proficiencies_for_objective_on_date(self, objective_id, from_, to):
        """Gets a ``ProficiencyList`` relating to the given objective effective during the entire given date range inclusive but not confined to the date range.

        arg:    objective_id (osid.id.Id): an objective ``Id``
        arg:    from (osid.calendaring.DateTime): starting date
        arg:    to (osid.calendaring.DateTime): ending date
        return: (osid.learning.ProficiencyList) - the returned
                ``Proficiency`` list
        raise:  InvalidArgument - ``from`` is greater than ``to``
        raise:  NullArgument - ``objective_id, from`` or ``to`` is
                ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.relationship.RelationshipLookupSession.get_relationships_for_destination_on_date
        proficiency_list = []
        for proficiency in self.get_proficiencies_for_objective():
            if overlap(from_, to, proficiency.start_date, proficiency.end_date):
                proficiency_list.append(proficiency)
        return objects.ProficiencyList(proficiency_list, runtime=self._runtime)