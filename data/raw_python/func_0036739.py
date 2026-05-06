def run_checks(self, service):
        """
        Runs each check for the service and reports to the service's discovery
        method based on the results.

        If all checks pass and the service's present node was previously
        reported as down, the present node is reported as up.  Conversely, if
        any of the checks fail and the service's present node was previously
        reported as up, the present node will be reported as down.
        """
        logger.debug("Running checks. (%s)", service.name)

        if service.discovery not in self.configurables[Discovery]:
            logger.warn(
                "Service %s is using Unknown/unavailable discovery '%s'.",
                service.name, service.discovery
            )
            return set(), set()

        service.update_ports()

        came_up, went_down = service.run_checks()

        return came_up, went_down