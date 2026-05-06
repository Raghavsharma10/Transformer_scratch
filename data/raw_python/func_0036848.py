def update_checks(self, check_configs):
        """
        Maintains the values in the `checks` attribute's dictionary.  Each
        key in the dictionary is a port, and each value is a nested dictionary
        mapping each check's name to the Check instance.

        This method makes sure the attribute reflects all of the properly
        configured checks and ports.  Removing no-longer-configured ports
        is left to the `run_checks` method.
        """
        for check_name, check_config in six.iteritems(check_configs):
            if check_name == "interval":
                continue

            for port in self.ports:
                try:
                    check = Check.from_config(check_name, check_config)
                    check.host = self.host
                    check.port = port
                    self.checks[port][check_name] = check
                except ValueError as e:
                    logger.error(
                        "Error when configuring check '%s' for service %s: %s",
                        check_name, self.name, str(e)
                    )
                    continue