def precision(self, precision):
        """
        For queries that should run faster, you may specify a lower precision,
        and for those that need to be more precise, a higher precision:

        ```python
        # faster queries
        query.range('2014-01-01', '2014-01-31', precision=0)
        query.range('2014-01-01', '2014-01-31', precision='FASTER')
        # queries with the default level of precision (usually what you want)
        query.range('2014-01-01', '2014-01-31')
        query.range('2014-01-01', '2014-01-31', precision=1)
        query.range('2014-01-01', '2014-01-31', precision='DEFAULT')
        # queries that are more precise
        query.range('2014-01-01', '2014-01-31', precision=2)
        query.range('2014-01-01', '2014-01-31', precision='HIGHER_PRECISION')
        ```
        """

        if isinstance(precision, int):
            precision = self.PRECISION_LEVELS[precision]

        if precision not in self.PRECISION_LEVELS:
            levels = ", ".join(self.PRECISION_LEVELS)
            raise ValueError("Precision should be one of: " + levels)

        if precision != 'DEFAULT':
            self.raw.update({'samplingLevel': precision})

        return self