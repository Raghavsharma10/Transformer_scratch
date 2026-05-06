def compare(self, other):
        """Compare the DigitWord with another DigitWord (other) and provided iterated analysis of the
        matches (none or loose) and the occurrence (one or more) of each DigitEntry in both
        DigitWords. The method returns a list of Comparison objects."""

        self._validate_compare_parameters(other=other)

        return_list = []
        for idx, digit in enumerate(other):
            dwa = DigitWordAnalysis(
                index=idx,
                digit=digit,
                match=(digit == self._word[idx]),
                in_word=(self._word.count(digit) > 0),
                multiple=(self._word.count(digit) > 1)
            )
            return_list.append(dwa)

        return return_list