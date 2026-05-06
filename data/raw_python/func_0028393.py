def addNotice(self, data):
        """
        Add custom notice to front-end for this NodeServers

        :param data: String of characters to add as a notification in the front-end.
        """
        LOGGER.info('Sending addnotice to Polyglot: {}'.format(data))
        message = { 'addnotice': data }
        self.send(message)