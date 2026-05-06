def get_state_in_ec_string(self, ec_index, add_colour=True):
        '''Get the state of the component in an execution context as a string.

        @param ec_index The index of the execution context to check the state
                        in. This index is into the total array of contexts,
                        that is both owned and participating contexts. If the
                        value of ec_index is greater than the length of @ref
                        owned_ecs, that length is subtracted from ec_index and
                        the result used as an index into @ref
                        participating_ecs.

        '''
        with self._mutex:
            if ec_index >= len(self.owned_ecs):
                ec_index -= len(self.owned_ecs)
                if ec_index >= len(self.participating_ecs):
                    raise exceptions.BadECIndexError(ec_index)
                state = self.participating_ec_states[ec_index]
            else:
                state = self.owned_ec_states[ec_index]
        if state == self.INACTIVE:
            result = 'Inactive', ['bold', 'blue']
        elif state == self.ACTIVE:
            result = 'Active', ['bold', 'green']
        elif state == self.ERROR:
            result = 'Error', ['bold', 'white', 'bgred']
        elif state == self.UNKNOWN:
            result = 'Unknown', ['bold', 'red']
        elif state == self.CREATED:
            result = 'Created', ['reset']
        if add_colour:
            return utils.build_attr_string(result[1], supported=add_colour) + \
                    result[0] + utils.build_attr_string('reset',
                    supported=add_colour)
        else:
            return result[0]