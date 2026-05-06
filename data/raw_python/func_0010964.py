def on_error(self, status_code):
        """Called when a non-200 status code is returned"""
        logger.error('Twitter returned error code %s', status_code)
        self.error = status_code
        return False