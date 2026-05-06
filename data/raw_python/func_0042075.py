def push_state(self):
        """
        Push a copy of the topmost state on top of the state stack,
        returns the new top.
        """

        new = dict(self.states[-1])
        self.states.append(new)
        return self.state