def state_in_ec(self, ec_index):
        '''Get the state of the component in an execution context.

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
                return self.participating_ec_states[ec_index]
            else:
                return self.owned_ec_states[ec_index]