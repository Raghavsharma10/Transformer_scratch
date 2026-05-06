def _generate_graph(self, name, title, stats_data, y_name):
        """
        Generate a downloads graph; append it to ``self._graphs``.

        :param name: HTML name of the graph, also used in ``self.GRAPH_KEYS``
        :type name: str
        :param title: human-readable title for the graph
        :type title: str
        :param stats_data: data dict from ``self._stats``
        :type stats_data: dict
        :param y_name: Y axis metric name
        :type y_name: str
        """
        logger.debug('Generating chart data for %s graph', name)
        orig_data, labels = self._data_dict_to_bokeh_chart_data(stats_data)
        data = self._limit_data(orig_data)
        logger.debug('Generating %s graph', name)
        script, div = FancyAreaGraph(
            name, '%s %s' % (self.project_name, title), data, labels,
            y_name).generate_graph()
        logger.debug('%s graph generated', name)
        self._graphs[name] = {
            'title': title,
            'script': script,
            'div': div,
            'raw_data': stats_data
        }