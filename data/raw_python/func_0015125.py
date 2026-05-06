def unsubscribe_multi(self, topics):
        """Unsubscribe to some topics."""
        if self.sock == NC.INVALID_SOCKET:
            return NC.ERR_NO_CONN
        
        self.logger.info("UNSUBSCRIBE: %s", ', '.join(topics))
        return self.send_unsubscribe(False, [utf8encode(topic) for topic in topics])