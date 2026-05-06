def owned_ec_states(self):
        '''The state of each execution context this component owns.'''
        with self._mutex:
            if not self._owned_ec_states:
                if self.owned_ecs:
                    states = []
                    for ec in self.owned_ecs:
                        states.append(self._get_ec_state(ec))
                    self._owned_ec_states = states
                else:
                    self._owned_ec_states = []
        return self._owned_ec_states