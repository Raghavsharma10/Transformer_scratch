def step(self, state, clamping):
        """
        Performs a simulation step from the given state and with respect to the given clamping

        Parameters
        ----------
        state : dict
            The key-value mapping describing the current state of the logical network

        clamping : caspo.core.clamping.Clamping
            A clamping over variables in the logical network

        Returns
        -------
        dict
            The key-value mapping describing the next state of the logical network
        """
        ns = state.copy()
        for var in state:
            if clamping.has_variable(var):
                ns[var] = int(clamping.bool(var))
            else:
                or_value = 0
                for clause, _ in self.in_edges_iter(var):
                    or_value = or_value or clause.bool(state)
                    if or_value:
                        break

                ns[var] = int(or_value)

        return ns