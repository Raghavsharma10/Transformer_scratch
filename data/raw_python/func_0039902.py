def iterate(self, state):
        """Process a starting state over and over again. Example:

        for x in rule_110.iterate(state):
            # Do something with the current state here
            # Note: You should break this yourself
        # This breaks automatically if the previous state was the same as the
        # current one, but that's not gonna happen on an infinite canvas
"""
        cur_state = state
        old_state = cur_state
        while True:
            cur_state = self.process(cur_state)
            if old_state == cur_state:
                break
            old_state = cur_state
            yield cur_state