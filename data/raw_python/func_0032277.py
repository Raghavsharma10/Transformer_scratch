def _periodic_task_iter(self):
        """
        Iterates through all the periodic tasks:

        - the service registry pinging
        - default dummy task if on Windows
        - user defined periodic tasks

        :return:
        """
        for strategy in self.discovery_strategies:
            self.default_periodic_tasks.append(
                (functools.partial(strategy.ping, self.name, self.accessible_at),
                 self.service_registry_ping_interval)
            )
            self.default_periodic_tasks[-1][0]()

        all_periodic_tasks = self.default_periodic_tasks + self.periodic_tasks
        for func, timer_in_seconds in all_periodic_tasks:
            timer_milisec = timer_in_seconds * 1000
            yield PeriodicCallback(func, timer_milisec, io_loop=self.io_loop)