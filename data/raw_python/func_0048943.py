def match_date(self, from_, to, match):
        """Matches temporals where the given date range falls entirely between the start and end dates inclusive.

        arg:    from (osid.calendaring.DateTime): start date
        arg:    to (osid.calendaring.DateTime): end date
        arg:    match (boolean): ``true`` if a positive match, ``false``
                for a negative match
        raise:  InvalidArgument - ``from`` is less than ``to``
        raise:  NullArgument - ``from`` or ``to`` is ``null``
        *compliance: mandatory -- This method must be implemented.*

        """
        if match:
            if to < from_:
                raise errors.InvalidArgument('end date must be >= start date when match = True')
            self._query_terms['startDate'] = {
                '$gte': from_
            }
            self._query_terms['endDate'] = {
                '$lte': to
            }
        else:
            raise errors.InvalidArgument('match = False not currently supported')