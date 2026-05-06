def disconnect(self):
        """
        Stops and closes the kazoo connection.
        """
        logger.info("Disconnecting from Zookeeper.")
        self.client.stop()
        self.client.close()