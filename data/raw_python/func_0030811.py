def dataframe(self, filtered_dims={}, unstack=False, df_class=None, add_code=False):
        """
        Yield rows in a reduced format, with one dimension as an index, one measure column per
        secondary dimension, and all other dimensions filtered.


        :param measure: The column names of one or more measures
        :param p_dim: The primary dimension. This will be the index of the dataframe.
        :param s_dim: a secondary dimension. The returned frame will be unstacked on this dimension
        :param unstack:
        :param filtered_dims: A dict of dimension columns names that are filtered, mapped to the dimension value
        to select.
        :param add_code: When substitution a label for a column, also add the code value.
        :return:
        """

        measure = self.table.column(measure)
        p_dim = self.table.column(p_dim)

        assert measure
        assert p_dim

        if s_dim:
            s_dim = self.table.column(s_dim)

        from six import text_type

        def maybe_quote(v):
            from six import string_types
            if isinstance(v, string_types):
                return '"{}"'.format(v)
            else:
                return v

        all_dims = [p_dim.name] + filtered_dims.keys()

        if s_dim:
            all_dims.append(s_dim.name)

        if filtered_dims:
            all_dims += filtered_dims.keys()

        all_dims = [text_type(c) for c in all_dims]

        # "primary_dimensions" means something different here, all of the dimensions in the
        # dataset that do not have children.
        primary_dims = [text_type(c.name) for c in self.primary_dimensions]

        if set(all_dims) != set(primary_dims):
            raise ValueError("The primary, secondary and filtered dimensions must cover all dimensions" +
                             " {} != {}".format(sorted(all_dims), sorted(primary_dims)))

        columns = []

        p_dim_label = None
        s_dim_label = None

        if p_dim.label:

            # For geographic datasets, also need the gvid
            if p_dim.type_is_gvid:
                columns.append(p_dim.name)

            p_dim = p_dim_label = p_dim.label
            columns.append(p_dim_label.name)



        else:
            columns.append(p_dim.name)

        if s_dim:

            if s_dim.label:
                s_dim = s_dim_label = s_dim.label
                columns.append(s_dim_label.name)
            else:
                columns.append(s_dim.name)

        columns.append(measure.name)

        # Create the predicate to filter out the filtered dimensions
        if filtered_dims:
            code = ' and '.join("row.{} == {}".format(k, maybe_quote(v)) for k, v in filtered_dims.items())

            predicate = eval('lambda row: {}'.format(code))
        else:
            predicate = lambda row: True

        df = self.analysis.dataframe(predicate, columns=columns, df_class=df_class)

        if unstack:
            # Need to set the s_dim in the index to get a hierarchical index, required for unstacking.
            # The final df will have only the p_dim as an index.

            if s_dim:
                df = df.set_index([p_dim.name, s_dim.name])

                df = df.unstack()

                df.columns = df.columns.get_level_values(1)  # [' '.join(col).strip() for col in df.columns.values]

            else:
                # Can't actually unstack without a second dimension.
                df = df.set_index(p_dim.name)

            df.reset_index()

        return df