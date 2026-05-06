def _loadHandlers(self):
        """
        creates a dictionary of named handler instances
        :return: the dictionary
        """
        return {handler.name: handler for handler in map(self.createHandler, self.config['handlers'])}