def publish_metric(self, metric_name, metric_value, epoch_seconds=None):
        '''Record a single hit on a given metric.

        Args:
            metric_name: The name of the metric to record with Carbon.
            metric_value: The value to record with Carbon.
            epoch_seconds: Optionally specify the time for the metric hit.

        Returns:
            None

        '''
        if epoch_seconds is None:
            epoch_seconds = self._reactor.seconds()
        self._client_factory.publish_metric(metric_name, metric_value, int(epoch_seconds))