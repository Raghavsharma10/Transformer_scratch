def match_start_date(self, start, end, match):
        """Matches temporals whose start date falls in between the given dates inclusive.

        arg:    start (osid.calendaring.DateTime): start of date range
        arg:    end (osid.calendaring.DateTime): end of date range
        arg:    match (boolean): ``true`` if a positive match, ``false``
                for a negative match
        raise:  InvalidArgument - ``start`` is less than ``end``
        raise:  NullArgument - ``start`` or ``end`` is ``null``
        *compliance: mandatory -- This method must be implemented.*

        """
        if match:
            if end < start:
                raise errors.InvalidArgument('end date must be >= start date when match = True')
            self._query_terms['startDate'] = {
                '$gte': start,
                '$lte': end
            }
        else:
            raise errors.InvalidArgument('match = False not currently supported')