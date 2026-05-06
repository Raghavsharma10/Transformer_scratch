def start(self):
        """ Restart the listener
        """
        if not event.contains(self.field, 'set', self.__validate):
            self.__create_event()