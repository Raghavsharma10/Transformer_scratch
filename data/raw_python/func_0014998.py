def clean_metric_name(self, metric_name):
        """
        Make sure the metric is free of control chars, spaces, tabs, etc.
        """
        if not self._clean_metric_name:
            return metric_name
        metric_name = str(metric_name)
        for _from, _to in self.cleaning_replacement_list:
            metric_name = metric_name.replace(_from, _to)
        return metric_name