def _clear_minimum_terms(self, match_key):
        """clears minimum match_key term values"""
        try:  # clear match = True case
            del self._query_terms[match_key]['$gte']
        except KeyError:
            pass
        try:  # clear match = False case
            del self._query_terms[match_key]['$lt']
        except KeyError:
            pass
        try:
            if self._query_terms[match_key] == {}:
                del self._query_terms[match_key]
        except KeyError:
            pass