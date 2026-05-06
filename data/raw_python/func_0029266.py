def add_result_handler(self, handler):
        """Register a new result handler.

        """
        self._result_handlers.append(handler)
        # Reset sorted handlers
        if self._sorted_handlers:
            self._sorted_handlers = None