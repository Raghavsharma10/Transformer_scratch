def dataframe(self, measure, p_dim, s_dim=None, filters={}, df_class=None):
        """
        Return a dataframe with a sumse of the columns of the partition, including a measure and one
        or two dimensions. FOr dimensions that have labels, the labels are included

        The returned dataframe will have extra properties to describe the conversion:

        * plot_axes: List of dimension names for the first and second axis
        * labels: THe names of the label columns for the axes
        * filtered: The `filters` dict
        * floating: The names of primary dimensions that are not axes nor filtered

        THere is also an iterator, `rows`, which returns the header and then all of the rows.

        :param measure: The column names of one or more measures
        :param p_dim: The primary dimension. This will be the index of the dataframe.
        :param s_dim: a secondary dimension. The returned frame will be unstacked on this dimension
        :param filters: A dict of column names, mapped to a column value, indicating rows to select. a
        row that passes the filter must have the values for all given rows; the entries are ANDED
        :param df_class:
        :return: a Dataframe, with extra properties
        """

        import numpy as np

        measure = self.measure(measure)

        p_dim = self.dimension(p_dim)

        assert p_dim

        if s_dim:
            s_dim = self.dimension(s_dim)

        columns = set([measure.name, p_dim.name])

        if p_dim.label:

            # For geographic datasets, also need the gvid
            if p_dim.geoid:
                columns.add(p_dim.geoid.name)

            columns.add(p_dim.label.name)

        if s_dim:

            columns.add(s_dim.name)

            if s_dim.label:
                columns.add(s_dim.label.name)

        def maybe_quote(v):
            from six import string_types
            if isinstance(v, string_types):
                return '"{}"'.format(v)
            else:
                return v

        # Create the predicate to filter out the filtered dimensions
        if filters:

            selected_filters = []

            for k, v in filters.items():
                if isinstance(v, dict):
                    # The filter is actually the whole set of possible options, so
                    # just select the first one
                    v = v.keys()[0]

                selected_filters.append("row.{} == {}".format(k, maybe_quote(v)))

            code = ' and '.join(selected_filters)

            predicate = eval('lambda row: {}'.format(code))
        else:
            code = None

            def predicate(row):
                return True


        df = self.analysis.dataframe(predicate, columns=columns, df_class=df_class)

        if df is None or df.empty or len(df) == 0:
            return None

        # So we can track how many records were aggregated into each output row
        df['_count'] = 1

        def aggregate_string(x):
            return ', '.join(set(str(e) for e in x))

        agg = {
            '_count': 'count',

        }

        for col_name in columns:
            c = self.column(col_name)

            # The primary and secondary dimensions are put into the index by groupby
            if c.name == p_dim.name or (s_dim and c.name == s_dim.name):
                continue

            # FIXME! This will only work if the child is only level from the parent. Should
            # have an acessor for the top level.
            if c.parent and (c.parent == p_dim.name or (s_dim and c.parent == s_dim.name)):
                continue

            if c.is_measure:
                agg[c.name] = np.mean

            if c.is_dimension:
                agg[c.name] = aggregate_string

        plot_axes = [p_dim.name]

        if s_dim:
            plot_axes.append(s_dim.name)

        df = df.groupby(list(columns - set([measure.name]))).agg(agg).reset_index()

        df._metadata = ['plot_axes', 'filtered', 'floating', 'labels', 'dimension_set', 'measure']

        df.plot_axes = [c for c in plot_axes]
        df.filtered = filters

        # Dimensions that are not specified as axes nor filtered
        df.floating = list(set(c.name for c in self.primary_dimensions) -
                           set(df.filtered.keys()) -
                           set(df.plot_axes))

        df.labels = [self.column(c).label.name if self.column(c).label else c for c in df.plot_axes]

        df.dimension_set = self.dimension_set(p_dim, s_dim=s_dim)

        df.measure = measure.name

        def rows(self):
            yield ['id'] + list(df.columns)

            for t in df.itertuples():
                yield list(t)

        # Really should not do this, but I don't want to re-build the dataframe with another
        # class
        df.__class__.rows = property(rows)



        return df