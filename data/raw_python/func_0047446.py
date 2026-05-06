def match_completion(self, start, end, match):
        """Sets the completion for this query to match completion percentages between the given range inclusive.

        arg:    start (decimal): start of range
        arg:    end (decimal): end of range
        arg:    match (boolean): ``true`` for a positive match,
                ``false`` for a negative match
        raise:  InvalidArgument - ``end`` is less than ``start``
        *compliance: mandatory -- This method must be implemented.*

        """
        try:
            start = float(start)
        except ValueError:
            raise errors.InvalidArgument('Invalid start value')
        try:
            end = float(end)
        except ValueError:
            raise errors.InvalidArgument('Invalid end value')
        if match:
            if end < start:
                raise errors.InvalidArgument('end value must be >= start value when match = True')
            self._query_terms['completion'] = {
                '$gte': start,
                '$lte': end
            }
        else:
            raise errors.InvalidArgument('match = False not currently supported')