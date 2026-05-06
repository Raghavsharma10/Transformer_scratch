def saveCustomData(self, data):
        """
        Send custom dictionary to Polyglot to save and be retrieved on startup.

        :param data: Dictionary of key value pairs to store in Polyglot database.
        """
        LOGGER.info('Sending customData to Polyglot.')
        message = { 'customdata': data }
        self.send(message)