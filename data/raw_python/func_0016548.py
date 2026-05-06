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

        self.logger.debug('Executing action: %s', action)

        reward = self.wrapped.execute(action)
        if reward:
            self.total_reward += reward
        self.steps += 1

        self.logger.debug('Reward received on this step: %.5f',
                          reward or 0)
        self.logger.debug('Average reward per step: %.5f',
                          self.total_reward / self.steps)

        return reward