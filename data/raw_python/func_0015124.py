def unsubscribe(self, topic):
        """Unsubscribe to some topic."""
        if self.sock == NC.INVALID_SOCKET:
            return NC.ERR_NO_CONN
        
        self.logger.info("UNSUBSCRIBE: %s", topic)
        return self.send_unsubscribe(False, [utf8encode(topic)])