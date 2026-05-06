def on_service_add(self, service):
        """
        When a new service is added, a worker thread is launched to
        periodically run the checks for that service.
        """
        self.launch_thread(service.name, self.check_loop, service)