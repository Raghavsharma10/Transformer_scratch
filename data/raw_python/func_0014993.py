def send(self, metric, value, timestamp=None, formatter=None):
        """
        Format a single metric/value pair, and send it to the graphite
        server.

        :param metric: name of the metric
        :type prefix: string
        :param value: value of the metric
        :type prefix: float or int
        :param timestmap: epoch time of the event
        :type prefix: float or int
        :param formatter: option non-default formatter
        :type prefix: callable

        .. code-block:: python

          >>> g = init()
          >>> g.send("metric", 54)

        .. code-block:: python

          >>> g = init()
          >>> g.send(metric="metricname", value=73)

        """
        if formatter is None:
            formatter = self.formatter
        message = formatter(metric, value, timestamp)
        message = self. _presend(message)
        return self._dispatch_send(message)