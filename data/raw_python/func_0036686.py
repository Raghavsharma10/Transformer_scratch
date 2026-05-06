def participating_ec_states(self):
        '''The state of each execution context this component is participating
        in.

        '''
        with self._mutex:
            if not self._participating_ec_states:
                if self.participating_ecs:
                    states = []
                    for ec in self.participating_ecs:
                        states.append(self._get_ec_state(ec))
                    self._participating_ec_states = states
                else:
                    self._participating_ec_states = []
        return self._participating_ec_states