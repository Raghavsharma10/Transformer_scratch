def refresh_state_in_ec(self, ec_index):
        '''Get the up-to-date state of the component in an execution context.

        This function will update the state, rather than using the cached
        value. This may take time, if the component is executing on a remote
        node.

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
                state = self._get_ec_state(self.participating_ecs[ec_index])
                self.participating_ec_states[ec_index] = state
            else:
                state = self._get_ec_state(self.owned_ecs[ec_index])
                self.owned_ec_states[ec_index] = state
            return state