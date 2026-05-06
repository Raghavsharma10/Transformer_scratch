def dimension_set(self, p_dim, s_dim=None, dimensions=None, extant=set()):
        """
        Return a dict that describes the combination of one or two dimensions, for a plot

        :param p_dim:
        :param s_dim:
        :param dimensions:
        :param extant:
        :return:
        """

        if not dimensions:
            dimensions = self.primary_dimensions

        key = p_dim.name

        if s_dim:
            key += '/' + s_dim.name

        # Ignore if the key already exists or the primary and secondary dims are the same
        if key in extant or p_dim == s_dim:
            return

        # Don't allow geography to be a secondary dimension. It must either be a primary dimension
        # ( to make a map ) or a filter, or a small-multiple
        if s_dim and s_dim.valuetype_class.is_geo():
            return

        extant.add(key)

        filtered = {}

        for d in dimensions:
            if d != p_dim and d != s_dim:
                filtered[d.name] = d.pstats.uvalues.keys()

        if p_dim.valuetype_class.is_time():
            value_type = 'time'
            chart_type = 'line'
        elif p_dim.valuetype_class.is_geo():
            value_type = 'geo'
            chart_type = 'map'
        else:
            value_type = 'general'
            chart_type = 'bar'

        return dict(
            key=key,
            p_dim=p_dim.name,
            p_dim_type=value_type,
            p_label=p_dim.label_or_self.name,
            s_dim=s_dim.name if s_dim else None,
            s_label=s_dim.label_or_self.name if s_dim else None,
            filters=filtered,
            chart_type=chart_type
        )