def _detect_delimiter(self):
        """
        Detects the field delimiter in the sample data.
        """
        candidate_value = ','
        candidate_count = 0
        for delimiter in UniversalCsvReader.delimiters:
            count = self._sample.count(delimiter)
            if count > candidate_count:
                candidate_value = delimiter
                candidate_count = count

        self._formatting_parameters['delimiter'] = candidate_value