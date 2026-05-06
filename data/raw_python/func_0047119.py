def _add_match(self, match_key, match_value):
        """Adds a match key/value"""
        if match_key is None:
            raise errors.NullArgument()
        self._query_terms[match_key] = str(match_key) + '=' + str(match_value)