def match_objective_id(self, objective_id, match):
        """Sets the objective ``Id`` for this query.

        arg:    objective_id (osid.id.Id): an objective ``Id``
        arg:    match (boolean): ``true`` for a positive match,
                ``false`` for a negative match
        raise:  NullArgument - ``objective_id`` is ``null``
        *compliance: mandatory -- This method must be implemented.*

        """
        if not isinstance(objective_id, Id):
            raise errors.InvalidArgument()
        self._add_match('objectiveId', str(objective_id), match)