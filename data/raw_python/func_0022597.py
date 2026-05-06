def get_metric_history_chart_data(self, slugs, since=None, granularity='daily'):
        """Provides the same data as ``get_metric_history``, but with metrics
        data arranged in a format that's easy to plot with Chart.js. If you had
        the following yearly history, for example::

            [
                ('m:bar:y:2012', '1'),
                ('m:bar:y:2013', '2'),
                ('m:bar:y:2014', '3'),
                ('m:foo:y:2012', '4'),
                ('m:foo:y:2013', '5')
                ('m:foo:y:2014', '6')
            ]

        this method would provide you with the following data structure::

            'periods': ['y:2012', 'y:2013', 'y:2014']
            'data': [
              {
                'slug': 'bar',
                'values': [1, 2, 3]
              },
              {
                'slug': 'foo',
                'values': [4, 5, 6]
              },
            ]

        """
        slugs = sorted(slugs)
        history = self.get_metric_history(slugs, since, granularity=granularity)

        # Convert the history into an intermediate data structure organized
        # by periods. Since the history is sorted by key (which includes both
        # the slug and the date, the values should be ordered correctly.
        periods = []
        data = OrderedDict()
        for k, v in history:
            period = template_tags.strip_metric_prefix(k)
            if period not in periods:
                periods.append(period)

            slug = template_tags.metric_slug(k)
            if slug not in data:
                data[slug] = []
            data[slug].append(v)

        # Now, reorganize data for our end result.
        metrics = {'periods': periods, 'data': []}
        for slug, values in data.items():
            metrics['data'].append({
                'slug': slug,
                'values': values
            })

        return metrics