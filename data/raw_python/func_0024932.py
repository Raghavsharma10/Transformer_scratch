def _health_check_thread(self):
        """
        Health checker thread that pings the service every 30 seconds
        :return: None
        """
        while self._run_health_checker:
            response = self._health_check(Health_pb2.HealthCheckRequest(service='predix-event-hub.grpc.health'))
            logging.debug('received health check: ' + str(response))
            time.sleep(30)
        return