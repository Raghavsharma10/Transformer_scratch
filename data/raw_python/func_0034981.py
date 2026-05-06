def _copy(self):
        """
        Create a new L{TransitionTable} just like this one using a copy of the
        underlying transition table.

        @rtype: L{TransitionTable}
        """
        table = {}
        for existingState, existingOutputs in self.table.items():
            table[existingState] = {}
            for (existingInput, existingTransition) in existingOutputs.items():
                table[existingState][existingInput] = existingTransition
        return TransitionTable(table)