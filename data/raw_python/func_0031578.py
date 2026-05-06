def add_matches(self, matcher, lhs,
                    match_params=[], include_params=[], exclude_params=[],
                    numq=1, flatten=None):
        """
        Quick way to call `add_or_matches` and `add_and_matches`.
        """
        matcher = adapt_matcher(matcher)
        notmatcher = negate(matcher)
        self.add_and_matches(matcher, lhs, match_params, numq, flatten)
        self.add_or_matches(matcher, lhs, include_params, numq, flatten)
        self.add_and_matches(notmatcher, lhs, exclude_params, numq, flatten)