def get_state_string(self, add_colour=True):
        '''Get the state of this component as an optionally-coloured string.

        @param add_colour If True, ANSI colour codes will be added to the
                          string.
        @return A string describing the state of this component.

        '''
        with self._mutex:
            if self.state == self.INACTIVE:
                result = 'Inactive', ['bold', 'blue']
            elif self.state == self.ACTIVE:
                result = 'Active', ['bold', 'green']
            elif self.state == self.ERROR:
                result = 'Error', ['bold', 'white', 'bgred']
            elif self.state == self.UNKNOWN:
                result = 'Unknown', ['bold', 'red']
            elif self.state == self.CREATED:
                result = 'Created', ['reset']
        if add_colour:
            return utils.build_attr_string(result[1], supported=add_colour) + \
                    result[0] + utils.build_attr_string('reset', supported=add_colour)
        else:
            return result[0]