def _data_dict_to_bokeh_chart_data(self, data):
        """
        Take a dictionary of data, as returned by the :py:class:`~.ProjectStats`
        per_*_data properties, return a 2-tuple of data dict and x labels list
        usable by bokeh.charts.

        :param data: data dict from :py:class:`~.ProjectStats` property
        :type data: dict
        :return: 2-tuple of data dict, x labels list
        :rtype: tuple
        """
        labels = []
        # find all the data keys
        keys = set()
        for date in data:
            for k in data[date]:
                keys.add(k)
        # final output dict
        out_data = {}
        for k in keys:
            out_data[k] = []
        # transform the data; deal with sparse data
        for data_date, data_dict in sorted(data.items()):
            labels.append(data_date)
            for k in out_data:
                if k in data_dict:
                    out_data[k].append(data_dict[k])
                else:
                    out_data[k].append(0)
        return out_data, labels