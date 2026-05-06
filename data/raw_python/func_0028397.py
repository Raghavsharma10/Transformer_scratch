def inConfig(self, config):
        """
        Save incoming config received from Polyglot to Interface.config and then do any functions
        that are waiting on the config to be received.
        """
        self.config = config
        self.isyVersion = config['isyVersion']
        try:
            for watcher in self.__configObservers:
                watcher(config)

            self.send_custom_config_docs()

        except KeyError as e:
            LOGGER.error('KeyError in gotConfig: {}'.format(e), exc_info=True)