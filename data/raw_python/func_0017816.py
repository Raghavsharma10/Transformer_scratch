def wait_for_web_available(self):
        """
        Wait for the web server to become available or raise DatacatsError
        if it fails to start.
        """
        try:
            if not wait_for_service_available(
                    self._get_container_name('web'),
                    self.web_address(),
                    WEB_START_TIMEOUT_SECONDS):
                raise DatacatsError('Error while starting web container:\n' +
                                    container_logs(self._get_container_name('web'), "all",
                                                   False, None))
        except ServiceTimeout:
            raise DatacatsError('Timeout while starting web container. Logs:' +
                                container_logs(self._get_container_name('web'), "all", False, None))