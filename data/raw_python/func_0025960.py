def createHandler(self, handler):
        """
        Creates a data handler from the input configuration.
        :param handler: the handler cfg.
        :return: the constructed handler.
        """
        target = handler['target']
        if handler['type'] == 'log':
            self.logger.warning("Initialising csvlogger to log data to " + target)
            return CSVLogger('recorder', handler['name'], target)
        elif handler['type'] == 'post':
            self.logger.warning("Initialising http logger to log data to " + target)
            return HttpPoster(handler['name'], target)