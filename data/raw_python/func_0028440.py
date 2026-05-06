def _after_request(self, response):
        """
        The signal handler for the request_finished signal.
        """
        if not getattr(g, '_has_exception', False):
            extra = self.summary_extra()
            self.summary_logger.info('', extra=extra)
        return response