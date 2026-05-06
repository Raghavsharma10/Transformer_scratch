def _clear_maximum_terms(self, match_key):
        """clears maximum match_key term values"""
        try:  # clear match = True case
            del self._query_terms[match_key]['$lte']
        except KeyError:
            pass
        try:  # clear match = False case
            del self._query_terms[match_key]['$gt']
        except KeyError:
            pass
        try:
            if self._query_terms[match_key] == {}:
                del self._query_terms[match_key]
        except KeyError:
            pass