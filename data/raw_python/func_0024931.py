def _init_health_checker(self):
        """
        start the health checker stub and start a thread to ping it every 30 seconds
        :return: None
        """
        stub = Health_pb2_grpc.HealthStub(channel=self._channel)
        self._health_check = stub.Check
        health_check_thread = threading.Thread(target=self._health_check_thread)
        health_check_thread.daemon = True
        health_check_thread.start()