def bool(self, state):
        """
        Returns the Boolean evaluation of the clause with respect to a given state

        Parameters
        ----------
        state : dict
            Key-value mapping describing a Boolean state or assignment

        Returns
        -------
        boolean
            The evaluation of the clause with respect to the given state or assignment
        """
        value = 1
        for source, sign in self:
            value = value and (state[source] if sign == 1 else not state[source])
            if not value:
                break

        return value