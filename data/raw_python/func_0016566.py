def run(self, scenario, learn=True):
        """Run the algorithm, utilizing the classifier set to choose the
        most appropriate action for each situation produced by the
        scenario. If learn is True, improve the situation/action mapping to
        maximize reward. Otherwise, ignore any reward received.

        Usage:
            model.run(scenario, learn=True)

        Arguments:
            scenario: A Scenario instance which this classifier set is to
                interact with.
            learn: A bool indicating whether the classifier set should
                attempt to optimize its performance based on reward
                received for each action, as opposed to simply using what
                it has already learned from previous runs and ignoring
                reward received; default is True.
        Return: None
        """

        assert isinstance(scenario, scenarios.Scenario)

        previous_match_set = None

        # Repeat until the scenario has run its course.
        while scenario.more():
            # Gather information about the current state of the
            # environment.
            situation = scenario.sense()

            # Determine which rules match the current situation.
            match_set = self.match(situation)

            # Select the best action for the current situation (or a random
            # one, if we are on an exploration step).
            match_set.select_action()

            # Perform the selected action
            # and find out what the received reward was.
            reward = scenario.execute(match_set.selected_action)

            # If the scenario is dynamic, don't immediately apply the
            # reward; instead, wait until the next iteration and factor in
            # not only the reward that was received on the previous step,
            # but the (discounted) reward that is expected going forward
            # given the resulting situation observed after the action was
            # taken. This is a classic feature of temporal difference (TD)
            # algorithms, which acts to stitch together a general picture
            # of the future expected reward without actually waiting the
            # full duration to find out what it will be.
            if learn:
                # Ensure we are not trying to learn in a non-learning
                # scenario.
                assert reward is not None

                if scenario.is_dynamic:
                    if previous_match_set is not None:
                        match_set.pay(previous_match_set)
                        previous_match_set.apply_payoff()
                    match_set.payoff = reward

                    # Remember the current reward and match set for the
                    # next iteration.
                    previous_match_set = match_set
                else:
                    match_set.payoff = reward
                    match_set.apply_payoff()

        # This serves to tie off the final stitch. The last action taken
        # gets only the immediate reward; there is no future reward
        # expected.
        if learn and previous_match_set is not None:
            previous_match_set.apply_payoff()