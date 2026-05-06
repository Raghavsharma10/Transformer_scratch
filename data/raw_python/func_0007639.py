def sort(self, *columns, **options):
        """
        Return a new query which will produce results sorted by
        one or more metrics or dimensions. You may use plain
        strings for the columns, or actual `Column`, `Metric`
        and `Dimension` objects.

        Add a minus in front of the metric (either the string or
        the object) to sort in descending order.

        ```python
        # sort using strings
        query.sort('pageviews', '-device type')
        # alternatively, ask for a descending sort in a keyword argument
        query.sort('pageviews', descending=True)

        # sort using metric, dimension or column objects
        pageviews = profile.core.metrics['pageviews']
        query.sort(-pageviews)
        ```
        """

        sorts = self.meta.setdefault('sort', [])

        for column in columns:
            if isinstance(column, Column):
                identifier = column.id
            elif isinstance(column, utils.basestring):
                descending = column.startswith('-') or options.get('descending', False)
                identifier = self.api.columns[column.lstrip('-')].id
            else:
                raise ValueError("Can only sort on columns or column strings. Received: {}".format(column))

            if descending:
                sign = '-'
            else:
                sign = ''

            sorts.append(sign + identifier)

        self.raw['sort'] = ",".join(sorts)
        return self