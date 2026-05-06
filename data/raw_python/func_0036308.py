def running_as_string(self, add_colour=True):
        '''Get the state of this context as an optionally coloured string.

        @param add_colour If True, ANSI colour codes will be added.
        @return A string describing this context's running state.

        '''
        with self._mutex:
            if self.running:
                result = 'Running', ['bold', 'green']
            else:
                result = 'Stopped', ['reset']
        if add_colour:
            return utils.build_attr_string(result[1], supported=add_colour) + \
                    result[0] + utils.build_attr_string('reset', supported=add_colour)
        else:
            return result[0]