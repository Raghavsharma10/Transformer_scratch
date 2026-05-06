def addTransition(self, state, input, output, nextState):
        """
        Create a new L{TransitionTable} with all the same transitions as this
        L{TransitionTable} plus a new transition.

        @param state: The state for which the new transition is defined.
        @param input: The input that triggers the new transition.
        @param output: The output produced by the new transition.
        @param nextState: The state that will follow the new transition.

        @return: The newly created L{TransitionTable}.
        """
        return self.addTransitions(state, {input: (output, nextState)})