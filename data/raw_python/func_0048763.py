def match_deadline(self, start, end, match):
        """Matches assessments whose end time falls between the specified range inclusive.

        arg:    start (osid.calendaring.DateTime): start of range
        arg:    end (osid.calendaring.DateTime): end of range
        arg:    match (boolean): ``true`` for a positive match,
                ``false`` for a negative match
        raise:  InvalidArgument - ``end`` is less than ``start``
        raise:  NullArgument - ``start`` or ``end`` is ``null``
        *compliance: mandatory -- This method must be implemented.*

        """
        self._match_minimum_date_time('deadline', start, match)
        self._match_maximum_date_time('deadline', end, match)