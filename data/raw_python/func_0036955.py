def on_balancer_remove(self, name):
        """
        The removal of a load balancer config isn't supported just yet.

        If the balancer being removed is the only configured one we fire
        a critical log message saying so.  A writer setup with no balancers
        is less than useless.
        """
        if len(self.configurables[Balancer]) == 1:
            logger.critical(
                "'%s' config file removed! It was the only balancer left!",
                name
            )