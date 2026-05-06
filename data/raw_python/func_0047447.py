def match_level_id(self, grade_id, match):
        """Sets the level grade ``Id`` for this query.

        arg:    grade_id (osid.id.Id): a grade ``Id``
        arg:    match (boolean): ``true`` for a positive match,
                ``false`` for a negative match
        raise:  NullArgument - ``grade_id`` is ``null``
        *compliance: mandatory -- This method must be implemented.*

        """
        if not isinstance(grade_id, Id):
            raise errors.InvalidArgument()
        self._add_match('levelId', str(grade_id), match)