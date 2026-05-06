def addTransitions(self, state, transitions):
        """
        Create a new L{TransitionTable} with all the same transitions as this
        L{TransitionTable} plus a number of new transitions.

        @param state: The state for which the new transitions are defined.
        @param transitions: A L{dict} mapping inputs to output, nextState
            pairs.  Each item from this L{dict} will define a new transition in
            C{state}.

        @return: The newly created L{TransitionTable}.
        """
        table = self._copy()
        state = table.table.setdefault(state, {})
        for (input, (output, nextState)) in transitions.items():
            state[input] = Transition(output, nextState)
        return table