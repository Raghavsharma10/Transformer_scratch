def send_dict(self, data, timestamp=None, formatter=None):
        """
        Format a dict of metric/values pairs, and send them all to the
        graphite server.

        :param data: key,value pair of metric name and metric value
        :type prefix: dict
        :param timestmap: epoch time of the event
        :type prefix: float or int
        :param formatter: option non-default formatter
        :type prefix: callable

        .. code-block:: python

          >>> g = init()
          >>> g.send_dict({'metric1': 54, 'metric2': 43, 'metricN': 999})

        """
        if formatter is None:
            formatter = self.formatter

        metric_list = []

        for metric, value in data.items():
            tmp_message = formatter(metric, value, timestamp)
            metric_list.append(tmp_message)

        message = "".join(metric_list)
        return self._dispatch_send(message)