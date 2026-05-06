def _filter_extracted(self, extracted_list):
        """Filter insignificant words for key noun phrase extraction.

        determiners, relative pronouns, reflexive pronouns
        In general, pronouns are not useful, as you need context to know what they refer to.
        Most of the pronouns, however, are filtered out by blob.noun_phrase method's
        np length (>1) filter

        :param list extracted_list: A list of noun phrases extracted from parser output.

        """
        _filtered = []
        for np in extracted_list:
            _np = np.split()
            if _np[0] in INSIGNIFICANT:
                _np.pop(0)
            try:
                if _np[-1] in INSIGNIFICANT:
                    _np.pop(-1)
                # e.g. 'welcher die ...'
                if _np[0] in INSIGNIFICANT:
                    _np.pop(0)
            except IndexError:
                _np = []
            if len(_np) > 0:
                _filtered.append(" ".join(_np))
        return _filtered