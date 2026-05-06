def insert_variables(
            self, device2variable, exchangespec, selections) -> None:
        """Determine the relevant target or base variables (as defined by
        the given |ExchangeSpecification| object ) handled by the given
        |Selections| object and insert them into the given `device2variable`
        dictionary."""
        if self.targetspecs.master in ('node', 'nodes'):
            for node in selections.nodes:
                variable = self._query_nodevariable(node, exchangespec)
                device2variable[node] = variable
        else:
            for element in self._iter_relevantelements(selections):
                variable = self._query_elementvariable(element, exchangespec)
                device2variable[element] = variable