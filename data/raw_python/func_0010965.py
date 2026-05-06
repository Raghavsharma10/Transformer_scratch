def on_exception(self, exception):
        """An exception occurred in the streaming thread"""
        logger.error('Exception from stream!', exc_info=True)
        self.streaming_exception = exception