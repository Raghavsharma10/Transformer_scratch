def _match_minimum_date_time(self, match_key, date_time_value, match=True):
        """Matches a minimum date time value"""
        if match:
            gtelt = '$gte'
        else:
            gtelt = '$lt'
        if match_key in self._query_terms:
            self._query_terms[match_key][gtelt] = date_time_value
        else:
            self._query_terms[match_key] = {gtelt: date_time_value}