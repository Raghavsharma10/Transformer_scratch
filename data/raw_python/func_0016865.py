def stop(self, stop_context):
        """ Perform any logic on solution stop """
        for p in self._providers:
            p.stop(stop_context)

        if self._clear_stop:
            self.clear_cache()