def on_disconnect(self, code, stream_name, reason):
        """Called when a disconnect is received"""
        logger.error('Disconnect message: %s %s %s', code, stream_name, reason)
        return True