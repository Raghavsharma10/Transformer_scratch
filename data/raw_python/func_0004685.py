def get(self, timeout=None):
        """Return result from the pipeline."""
        result = None
        for stage in self._output_stages:
            result = stage.get(timeout)
        return result