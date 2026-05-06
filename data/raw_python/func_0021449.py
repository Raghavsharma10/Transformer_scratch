def __create_event(self):
        """ Create an SQLAlchemy event listening the 'set' in a particular column.

        :rtype : object
        """
        if not event.contains(self.field, 'set', self.__validate):
            event.listen(self.field, 'set', self.__validate, retval=True)