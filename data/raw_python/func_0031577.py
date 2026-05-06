def add_and_matches(self, matcher, lhs, params, numq=1, flatten=None):
        """
        Add AND conditions to match to `params`.

        :type matcher: str or callable
        :arg  matcher: if `str`, `matcher.format` is used.
        :type     lhs: str
        :arg      lhs: the first argument to `matcher`.
        :type  params: list
        :arg   params: each element should be able to feed into sqlite '?'.
        :type    numq: int
        :arg     numq: number of parameters for each condition.
        :type flatten: None or callable
        :arg  flatten: when `numq > 1`, it should return a list of
                       length `numq * len(params)`.

        """
        params = self._adapt_params(params)
        qs = ['?'] * numq
        flatten = flatten or self._default_flatten(numq)
        expr = repeat(adapt_matcher(matcher)(lhs, *qs), len(params))
        self.conditions.extend(expr)
        self.params.extend(flatten(params))