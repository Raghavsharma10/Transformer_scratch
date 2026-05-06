def _norm_include(self, record, hist=None):
        """
        Normalization 'normIncludes' replace 'almost' values based on at least
        one of the following: includes strings, excludes strings, starts with
        string, ends with string

        :param dict record: dictionary of values to validate
        :param dict hist: existing input of history values
        """

        if hist is None:
            hist = {}

        for field in record:

            if record[field] != '' and record[field] is not None:

                if field in self.fields:

                    if 'normIncludes' in self.fields[field]['lookup']:

                        field_val_new, hist, _ = IncludesLookup(
                            fieldVal=record[field],
                            lookupType='normIncludes',
                            db=self.mongo,
                            fieldName=field,
                            histObj=hist)

                        record[field] = field_val_new

        return record, hist