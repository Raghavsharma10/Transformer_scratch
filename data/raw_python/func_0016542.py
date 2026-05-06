def execute(self, action):
        """Execute the indicated action within the environment and
        return the resulting immediate reward dictated by the reward
        program.

        Usage:
            immediate_reward = scenario.execute(selected_action)

        Arguments:
            action: The action to be executed within the current situation.
        Return:
            A float, the reward received for the action that was executed,
            or None if no reward is offered.
        """

        assert action in self.possible_actions

        self.remaining_cycles -= 1
        index = int(bitstrings.BitString(
            self.current_situation[:self.address_size]
        ))
        bit = self.current_situation[self.address_size + index]
        return action == bit