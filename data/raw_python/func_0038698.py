def match_name(self, in_string, fuzzy=False):
        """Match a color to a sRGB value.

        The matching will be based purely on the input string and the color names in the
        registry. If there's no direct hit, a fuzzy matching algorithm is applied. This method
        will never fail to return a sRGB value, but depending on the score, it might or might
        not be a sensible result – as a rule of thumb, any score less then 90 indicates that
        there's a lot of guessing going on. It's the callers responsibility to judge if the return
        value should be trusted.

        In normalization terms, this method implements "normalize an arbitrary color name
        to a sRGB value".

        Args:
          in_string (string): The input string containing something resembling
            a color name.
          fuzzy (bool, optional): Try fuzzy matching if no exact match was found.
            Defaults to ``False``.

        Returns:
          A named tuple with the members `hex_code` and `score`.

        Raises:
          ValueError: If ``fuzzy`` is ``False`` and no match is found

        Examples:
          >>> tint_registry = TintRegistry()
          >>> tint_registry.match_name("rather white", fuzzy=True)
          MatchResult(hex_code=u'ffffff', score=95)

        """
        in_string = _normalize(in_string)
        if in_string in self._hex_by_color:
            return MatchResult(self._hex_by_color[in_string], 100)

        if not fuzzy:
            raise ValueError("No match for %r found." % in_string)

        # We want the standard scorer *plus* the set scorer, because colors are often
        # (but not always) related by sub-strings
        color_names = self._hex_by_color.keys()
        set_match = dict(fuzzywuzzy.process.extract(
            in_string,
            color_names,
            scorer=fuzzywuzzy.fuzz.token_set_ratio
        ))
        standard_match = dict(fuzzywuzzy.process.extract(in_string, color_names))

        # This would be much easier with a collections.Counter, but alas! it's a 2.7 feature.
        key_union = set(set_match) | set(standard_match)
        counter = ((n, set_match.get(n, 0) + standard_match.get(n, 0)) for n in key_union)
        color_name, score = sorted(counter, key=operator.itemgetter(1))[-1]

        return MatchResult(self._hex_by_color[color_name], score / 2)