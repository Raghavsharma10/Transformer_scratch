def check_loop(self, service):
        """
        While the reporter is not shutting down and the service being checked
        is present in the reporter's configuration, this method will launch a
        job to run all of the service's checks and then pause for the
        configured interval.
        """
        logger.info("Starting check loop for service '%s'", service.name)

        def handle_checks_result(f):
            try:
                came_up, went_down = f.result()
            except Exception:
                logger.exception("Error checking service '%s'", service.name)
                return

            if not came_up and not went_down:
                return

            discovery = self.configurables[Discovery][service.discovery]

            for port in came_up:
                logger.debug("Reporting %s, port %d up", service.name, port)
                discovery.report_up(service, port)
            for port in went_down:
                logger.debug("Reporting %s, port %d down", service.name, port)
                discovery.report_down(service, port)

        while (
                service in self.configurables[Service].values() and
                not self.shutdown.is_set()
        ):
            self.work_pool.submit(
                self.run_checks, service
            ).add_done_callback(
                handle_checks_result
            )

            logger.debug("sleeping for %s seconds", service.check_interval)
            wait_on_event(self.shutdown, timeout=service.check_interval)