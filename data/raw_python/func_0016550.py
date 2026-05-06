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
        reward = self.reward_function(
            action,
            self.classifications[self.steps]
        )
        self.total_reward += reward
        self.steps += 1
        return reward