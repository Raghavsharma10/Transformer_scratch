def stop(self):
        """ Remove the listener to stop the validation
        """
        if event.contains(self.field, 'set', self.__validate):
            event.remove(self.field, 'set', self.__validate)