def process_signal(self, signum):
        """Invoked whenever a signal is added to the stack.

        :param int signum: The signal that was added

        """
        if signum == signal.SIGTERM:
            LOGGER.info('Received SIGTERM, initiating shutdown')
            self.stop()
        elif signum == signal.SIGHUP:
            LOGGER.info('Received SIGHUP')
            if self.config.reload():
                LOGGER.info('Configuration reloaded')
                logging.config.dictConfig(self.config.logging)
                self.on_configuration_reloaded()
        elif signum == signal.SIGUSR1:
            self.on_sigusr1()
        elif signum == signal.SIGUSR2:
            self.on_sigusr2()