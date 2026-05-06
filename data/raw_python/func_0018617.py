def text_ratio(self):
        """Return a measure of the sequences' word similarity (float in [0,1]).

        Each word has weight equal to its length for this measure

        >>> m = WordMatcher(a=['abcdef', '12'], b=['abcdef', '34']) # 3/4 of the text is the same
        >>> '%.3f' % m.ratio() # normal ratio fails
        '0.500'
        >>> '%.3f' % m.text_ratio() # text ratio is accurate
        '0.750'
        """
        return _calculate_ratio(
            self.match_length(),
            self._text_length(self.a) + self._text_length(self.b),
        )