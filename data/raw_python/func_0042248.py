def run(self, record, hist=None):
        """
        By passing the input record to be cleaned, and returns the input record
        after cleaning with history(dictionary).

        :param dict record: dictionary of values to validate
        :param dict hist: existing input of history values
        :return:
        """

        if hist is None:
            hist = {}

        if record:
            # Run user-defined functions for beforeGenericValLookup
            record, hist = self._apply_udfs(record=record,
                                            hist=hist,
                                            udf_type='beforeGenericValLookup')

            # Run generic validation lookup
            record, hist = self._val_g_lookup(record=record, hist=hist)

            # Run user-defined functions for beforeGenericValRegex
            record, hist = self._apply_udfs(record=record,
                                            hist=hist,
                                            udf_type='beforeGenericValRegex')

            # Run generic validation regex
            record, hist = self._val_g_regex(record=record, hist=hist)

            # Run user-defined functions for beforeFieldSpecificLookup
            record, hist = self._apply_udfs(record=record,
                                            hist=hist,
                                            udf_type='beforeFieldSpecificLookup')

            # Run field-specific validation lookup
            record, hist = self._val_fs_lookup(record=record, hist=hist)

            # Run user-defined functions for beforeFieldSpecificLookup
            record, hist = self._apply_udfs(record=record,
                                            hist=hist,
                                            udf_type='beforeFieldSpecificRegex')

            # Run field-specific validation regex
            record, hist = self._val_fs_regex(record=record, hist=hist)

            # Run user-defined functions for beforeNormLookup
            record, hist = self._apply_udfs(record=record,
                                            hist=hist,
                                            udf_type='beforeNormLookup')

            # Run normalization lookup
            record, hist = self._norm_lookup(record=record, hist=hist)

            # Run user-defined functions for beforeNormRegex
            record, hist = self._apply_udfs(record=record,
                                            hist=hist,
                                            udf_type='beforeNormRegex')

            # Run normalization regex
            record, hist = self._norm_regex(record=record, hist=hist)

            # Run user-defined functions for beforeNormIncludes
            record, hist = self._apply_udfs(record=record,
                                            hist=hist,
                                            udf_type='beforeNormIncludes')

            # Run normalization includes
            record, hist = self._norm_include(record=record, hist=hist)

            # Run user-defined functions for beforeNormIncludes
            record, hist = self._apply_udfs(record=record,
                                            hist=hist,
                                            udf_type='beforeDerive')

            # Fill gaps / refresh derived data
            record, hist = self._derive(record=record, hist=hist)

            # Run user-defined functions for beforeNormIncludes
            record, hist = self._apply_udfs(record=record,
                                            hist=hist,
                                            udf_type='afterAll')

            return record, hist

        return None, hist