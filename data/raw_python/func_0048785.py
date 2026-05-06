def match_descendant_objective_id(self, objective_id=None, match=None):
        """Sets the objective ``Id`` for this query to match objectives that have the specified objective as a descendant.

        arg:    objective_id (osid.id.Id): an objective ``Id``
        arg:    match (boolean): ``true`` for a positive match,
                ``false`` for a negative match
        raise:  NullArgument - ``objective_id`` is ``null``
        *compliance: mandatory -- This method must be implemented.*

        """
        if match:
            self._add_match('descendantObjectiveId', objective_id)
        else:
            raise errors.Unimplemented()