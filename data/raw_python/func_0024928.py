def shutdown(self):
        """
        Shutdown the client, shutdown the sub clients and stop the health checker
        :return: None
        """
        self._run_health_checker = False
        if self.publisher is not None:
            self.publisher.shutdown()

        if self.subscriber is not None:
            self.subscriber.shutdown()