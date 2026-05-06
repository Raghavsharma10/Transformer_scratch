def start_supporting_containers(self, log_syslog=False):
        """
        Start all supporting containers (containers required for CKAN to
        operate) if they aren't already running.

            :param log_syslog: A flag to redirect all container logs to host's syslog

        """
        log_syslog = True if self.always_prod else log_syslog
        # in production we always use log_syslog driver (to aggregate all the logs)
        task.start_supporting_containers(
            self.sitedir,
            self.target,
            self.passwords,
            self._get_container_name,
            self.extra_containers,
            log_syslog=log_syslog
            )