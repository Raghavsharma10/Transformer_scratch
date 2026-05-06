def match_learning_objective_id(self, objective_id, match):
        """Sets the learning objective ``Id`` for this query.

        arg:    objective_id (osid.id.Id): a learning objective ``Id``
        arg:    match (boolean): ``true`` for a positive match,
                ``false`` for negative match
        raise:  NullArgument - ``objective_id`` is ``null``
        *compliance: mandatory -- This method must be implemented.*

        """
        self._add_match('learningObjectiveIds', str(objective_id), bool(match))