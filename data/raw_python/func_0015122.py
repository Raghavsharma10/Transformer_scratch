def subscribe(self, topic, qos):
        """Subscribe to some topic."""
        if self.sock == NC.INVALID_SOCKET:
            return NC.ERR_NO_CONN
        
        self.logger.info("SUBSCRIBE: %s", topic)
        return self.send_subscribe(False, [(utf8encode(topic), qos)])