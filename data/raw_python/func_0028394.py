def removeNotice(self, data):
        """
        Add custom notice to front-end for this NodeServers

        :param data: Index of notices list to remove.
        """
        LOGGER.info('Sending removenotice to Polyglot for index {}'.format(data))
        message = { 'removenotice': data }
        self.send(message)