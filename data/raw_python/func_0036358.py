def polarity_as_string(self, add_colour=True):
        '''Get the polarity of this interface as a string.

        @param add_colour If True, ANSI colour codes will be added to the
                          string.
        @return A string describing the polarity of this interface.

        '''
        with self._mutex:
            if self.polarity == self.PROVIDED:
                result = 'Provided', ['reset']
            elif self.polarity == self.REQUIRED:
                result = 'Required', ['reset']
            if add_colour:
                return utils.build_attr_string(result[1], supported=add_colour) + \
                        result[0] + utils.build_attr_string('reset',
                                supported=add_colour)
            else:
                return result[0]