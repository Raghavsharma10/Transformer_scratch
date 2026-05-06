def _format_params(self, type_, params):
        """Reformat some of the parameters for sapi."""
        if 'initial_state' in params:
            # NB: at this moment the error raised when initial_state does not match lin/quad (in
            # active qubits) is not very informative, but there is also no clean way to check here
            # that they match because lin can be either a list or a dict. In the future it would be
            # good to check.
            initial_state = params['initial_state']
            if isinstance(initial_state, Mapping):

                initial_state_list = [3]*self.properties['num_qubits']

                low = -1 if type_ == 'ising' else 0

                for v, val in initial_state.items():
                    if val == 3:
                        continue
                    if val <= 0:
                        initial_state_list[v] = low
                    else:
                        initial_state_list[v] = 1

                params['initial_state'] = initial_state_list