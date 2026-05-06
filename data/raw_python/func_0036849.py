def run_checks(self):
        """
        Iterates over the configured ports and runs the checks on each one.

        Returns a two-element tuple: the first is the set of ports that
        transitioned from down to up, the second is the set of ports that
        transitioned from up to down.

        Also handles the case where a check for a since-removed port is run,
        marking the port as down regardless of the check's result and removing
        the check(s) for the port.
        """
        came_up = set()
        went_down = set()

        for port in self.ports:
            checks = self.checks[port].values()

            if not checks:
                logger.warn("No checks defined for self: %s", self.name)

            for check in checks:
                check.run()

            checks_pass = all([check.passing for check in checks])

            if self.is_up[port] in (False, None) and checks_pass:
                came_up.add(port)
                self.is_up[port] = True
            elif self.is_up[port] in (True, None) and not checks_pass:
                went_down.add(port)
                self.is_up[port] = False

        for unused_port in set(self.checks.keys()) - self.ports:
            went_down.add(unused_port)
            del self.checks[unused_port]

        return came_up, went_down