def _detect_line_ending(self):
        """
        Detects the line ending in the sample data.
        """
        candidate_value = '\n'
        candidate_count = 0
        for line_ending in UniversalCsvReader.line_endings:
            count = self._sample.count(line_ending)
            if count > candidate_count:
                candidate_value = line_ending
                candidate_count = count

        self._formatting_parameters['line_terminator'] = candidate_value