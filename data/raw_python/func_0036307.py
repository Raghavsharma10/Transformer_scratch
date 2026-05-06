def kind_as_string(self, add_colour=True):
        '''Get the type of this context as an optionally coloured string.

        @param add_colour If True, ANSI colour codes will be added.
        @return A string describing the kind of execution context this is.

        '''
        with self._mutex:
            if self.kind == self.PERIODIC:
                result = 'Periodic', ['reset']
            elif self.kind == self.EVENT_DRIVEN:
                result = 'Event-driven', ['reset']
            elif self.kind == self.OTHER:
                result = 'Other', ['reset']
        if add_colour:
            return utils.build_attr_string(result[1], supported=add_colour) + \
                    result[0] + utils.build_attr_string('reset', supported=add_colour)
        else:
            return result[0]