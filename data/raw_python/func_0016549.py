def more(self):
        """Return a Boolean indicating whether additional actions may be
        executed, per the reward program.

        Usage:
            while scenario.more():
                situation = scenario.sense()
                selected_action = choice(possible_actions)
                reward = scenario.execute(selected_action)

        Arguments: None
        Return:
            A bool indicating whether additional situations remain in the
            current run.
        """
        more = self.wrapped.more()

        if not self.steps % 100:
            self.logger.info('Steps completed: %d', self.steps)
            self.logger.info('Average reward per step: %.5f',
                             self.total_reward / (self.steps or 1))
        if not more:
            self.logger.info('Run completed.')
            self.logger.info('Total steps: %d', self.steps)
            self.logger.info('Total reward received: %.5f',
                             self.total_reward)
            self.logger.info('Average reward per step: %.5f',
                             self.total_reward / (self.steps or 1))

        return more