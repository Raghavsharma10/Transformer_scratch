def call_plugins(self, step):
        '''
        For each plugins, check if a "step" method exist on it, and call it

        Args:
            step (str): The method to search and call on each plugin
        '''
        for plugin in self.plugins:
            try:
                getattr(plugin, step)()
            except AttributeError:
                self.logger.debug("{} doesn't exist on plugin {}".format(step, plugin))
            except TypeError:
                self.logger.debug("{} on plugin {} is not callable".format(step, plugin))