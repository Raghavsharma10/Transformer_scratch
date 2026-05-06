def _start_drone(self):
        """
        Restarts the drone
        """

        with open(self.config_file, 'r') as config_file:
            self.config = json.load(config_file, object_hook=asciify)

        mode = None
        if self.config['general']['mode'] == '' or self.config['general']['mode'] is None:
            logger.info('Drone has not been configured, awaiting configuration from Beeswarm server.')
        elif self.config['general']['mode'] == 'honeypot':
            mode = Honeypot
        elif self.config['general']['mode'] == 'client':
            mode = Client

        if mode:
            self.drone = mode(self.work_dir, self.config)
            self.drone_greenlet = gevent.spawn(self.drone.start)
            self.drone_greenlet.link_exception(self.on_exception)
            logger.info('Drone configured and running. ({0})'.format(self.id))