def register_repeating_metric(self, metric_name, frequency, getter):
        '''Record hits to a metric at a specified interval.

        Args:
            metric_name: The name of the metric to record with Carbon.
            frequency: The frequency with which to poll the getter and record the value with Carbon.
            getter: A function which takes no arguments and returns the value to record with Carbon.

        Returns:
            RepeatingMetricHandle instance. Call .stop() on it to stop recording the metric.

        '''
        l = task.LoopingCall(self._publish_repeating_metric, metric_name, getter)
        repeating_metric_handle = RepeatingMetricHandle(l, frequency)
        self._repeating_metric_handles.append(repeating_metric_handle)
        if self.running:
            repeating_metric_handle.start()
        return repeating_metric_handle