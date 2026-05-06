def distribute_payoff(self, match_set):
        """Distribute the payoff received in response to the selected
        action of the given match set among the rules in the action set
        which deserve credit for recommending the action. The match_set
        argument is the MatchSet instance which suggested the selected
        action and earned the payoff.

        Usage:
            match_set = model.match(situation)
            match_set.select_action()
            match_set.payoff = reward
            model.algorithm.distribute_payoff(match_set)

        Arguments:
            match_set: A MatchSet instance for which the accumulated payoff
                needs to be distributed among its classifier rules.
        Return: None
        """

        assert isinstance(match_set, MatchSet)
        assert match_set.algorithm is self
        assert match_set.selected_action is not None

        payoff = float(match_set.payoff)

        action_set = match_set[match_set.selected_action]
        action_set_size = sum(rule.numerosity for rule in action_set)

        # Update the average reward, error, and action set size of each
        # rule participating in the action set.
        for rule in action_set:
            rule.experience += 1

            update_rate = max(self.learning_rate, 1 / rule.experience)

            rule.average_reward += (
                (payoff - rule.average_reward) *
                update_rate
            )

            rule.error += (
                (abs(payoff - rule.average_reward) - rule.error) *
                update_rate

            )

            rule.action_set_size += (
                (action_set_size - rule.action_set_size) *
                update_rate
            )

        # Update the fitness of the rules.
        self._update_fitness(action_set)

        # If the parameters so indicate, perform action set subsumption.
        if self.do_action_set_subsumption:
            self._action_set_subsumption(action_set)