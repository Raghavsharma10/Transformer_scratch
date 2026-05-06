def get_metric_history_as_columns(self, slugs, since=None,
                                      granularity='daily'):
        """Provides the same data as ``get_metric_history``, but in a columnar
        format. If you had the following yearly history, for example::

            [
                ('m:bar:y:2012', '1'),
                ('m:bar:y:2013', '2'),
                ('m:foo:y:2012', '3'),
                ('m:foo:y:2013', '4')
            ]

        this method would provide you with the following data structure::

            [
                ['Period',  'bar',  'foo']
                ['y:2012',  '1',    '3'],
                ['y:2013',  '2',    '4'],
            ]

        Note that this also includes a header column. Data in this format may
        be useful for certain graphing libraries (I'm looking at you Google
        Charts LineChart).

        """
        history = self.get_metric_history(slugs, since, granularity=granularity)
        _history = []  # new, columnar history
        periods = ['Period']  # A separate, single column for the time period
        for s in slugs:
            column = [s]  # story all the data for a single slug
            for key, value in history:
                # ``metric_slug`` extracts the slug from the Redis Key
                if template_tags.metric_slug(key) == s:
                    column.append(value)

                # Get time period value as first column; This value is
                # duplicated in the Redis key for each value, so this is a bit
                # inefficient, but... oh well.
                period = template_tags.strip_metric_prefix(key)
                if period not in periods:
                    periods.append(period)

            _history.append(column)  # Remember that slug's column of data

        # Finally, stick the time periods in the first column.
        _history.insert(0, periods)
        return list(zip(*_history))