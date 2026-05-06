def range(self, start=None, stop=None, months=0, days=0):
        """
        Return a new query that fetches metrics within a certain date range.

        ```python
        query.range('2014-01-01', '2014-06-30')
        ```

        If you don't specify a `stop` argument, the date range will end today. If instead
        you meant to fetch just a single day's results, try:

        ```python
        query.range('2014-01-01', days=1)
        ```

        More generally, you can specify that you'd like a certain number of days,
        starting from a certain date:

        ```python
        query.range('2014-01-01', months=3)
        query.range('2014-01-01', days=28)
        ```

        Note that if you don't specify a granularity (either through the `interval`
        method or through the `hourly`, `daily`, `weekly`, `monthly` or `yearly`
        shortcut methods) you will get only a single result, encompassing the
        entire date range, per metric.

        **Note:** it is currently not possible to easily specify that you'd like
        to query the last last full week(s), month(s) et cetera.
        This will be added sometime in the future.
        """

        start, stop = utils.date.range(start, stop, months, days)

        self.raw.update({
            'start_date': start,
            'end_date': stop,
        })

        return self