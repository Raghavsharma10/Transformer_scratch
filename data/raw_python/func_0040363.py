def configure(self, config):
        """
        Configure Monitor, pull list of what to monitor, initialize threads
        """
        self.config = config
        self.update_monitors()

        # initialize thread pools
        for profile in ('worker', 'result'):
            for _ in range(config['threads'][profile]['number']):
                worker = threading.Thread(target=config['threads'][profile]['function'])
                worker.daemon = True
                worker.start()

        # send a heartbeat right away
        self.heartbeat()

        # setup interval jobs
        self.refresh_stopper = set_interval(config['interval']['refresh']*1000,
                                            self.update_monitors)
        self.heartbeat_stopper = set_interval(config['interval']['heartbeat']*1000,
                                              self.heartbeat)
        self.reporting_stopper = set_interval(config['interval']['reporting']*1000,
                                              self.reporting)

        return self