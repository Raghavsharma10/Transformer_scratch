def match_any(self, match):
        """Matches any object.

        arg:    match (boolean): ``true`` to match any object ``,``
                ``false`` to match no objects
        *compliance: mandatory -- This method must be implemented.*

        """
        match_key = '_id'
        param = '$exists'
        if match:
            flag = 'true'
        else:
            flag = 'false'
        if match_key in self._query_terms:
            self._query_terms[match_key][param] = flag
        else:
            self._query_terms[match_key] = {param: flag}