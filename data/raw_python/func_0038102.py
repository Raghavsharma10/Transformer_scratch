def _resolve_responses(self):
        """
        _resolve_responses
        """
        while True:
            message = self.res_queue.get()
            if message is None:
                _logger.debug("_resolve_responses thread is terminated")
                return
            self.__resolve_responses(message)