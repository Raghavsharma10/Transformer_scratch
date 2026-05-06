def match_grade_system_id(self, grade_system_id, match):
        """Sets the grade system ``Id`` for this query.

        arg:    grade_system_id (osid.id.Id): a grade system ``Id``
        arg:    match (boolean): ``true`` for a positive match,
                ``false`` for a negative match
        raise:  NullArgument - ``grade_system_id`` is ``null``
        *compliance: mandatory -- This method must be implemented.*

        """
        self._add_match('gradeSystemId', str(grade_system_id), bool(match))