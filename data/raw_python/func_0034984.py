def addTerminalState(self, state):
        """
        Create a new L{TransitionTable} with all of the same transitions as
        this L{TransitionTable} plus a new state with no transitions.

        @param state: The new state to include in the new table.

        @return: The newly created L{TransitionTable}.
        """
        table = self._copy()
        table.table[state] = {}
        return table