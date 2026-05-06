def get_proficiencies_on_date(self, from_, to):
        """Gets a ``ProficiencyList`` effecyive during the entire given date range inclusive but not confined to the date range.

        arg:    from (osid.calendaring.DateTime): starting date
        arg:    to (osid.calendaring.DateTime): ending date
        return: (osid.learning.ProficiencyList) - the returned
                ``Proficiency`` list
        raise:  InvalidArgument - ``from`` is greater than ``to``
        raise:  NullArgument - ``from`` or ``to`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.relationship.RelationshipLookupSession.get_relationships_on_date
        proficiency_list = []
        for proficiency in self.get_proficiencies():
            if overlap(from_, to, proficiency.start_date, proficiency.end_date):
                proficiency_list.append(proficiency)
        return objects.ProficiencyList(proficiency_list, runtime=self._runtime)