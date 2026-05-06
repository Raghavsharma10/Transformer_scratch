def update(self, match_set):
        """Update the classifier set from which the match set was drawn,
        e.g. by applying a genetic algorithm. The match_set argument is the
        MatchSet instance whose classifier set should be updated.

        Usage:
            match_set = model.match(situation)
            match_set.select_action()
            match_set.payoff = reward
            model.algorithm.distribute_payoff(match_set)
            model.algorithm.update(match_set)

        Arguments:
            match_set: A MatchSet instance for which the classifier set
                from which it was drawn needs to be updated based on the
                match set's payoff distribution.
        Return: None
        """

        assert isinstance(match_set, MatchSet)
        assert match_set.model.algorithm is self
        assert match_set.selected_action is not None

        # Increment the iteration counter.
        match_set.model.update_time_stamp()

        action_set = match_set[match_set.selected_action]

        # If the average number of iterations since the last update for
        # each rule in the action set is too small, return early instead of
        # applying the GA.
        average_time_passed = (
            match_set.model.time_stamp -
            self._get_average_time_stamp(action_set)
        )
        if average_time_passed <= self.ga_threshold:
            return

        # Update the time step for each rule to indicate that they were
        # updated by the GA.
        self._set_timestamps(action_set)

        # Select two parents from the action set, with probability
        # proportionate to their fitness.
        parent1 = self._select_parent(action_set)
        parent2 = self._select_parent(action_set)

        # With the probability specified in the parameters, apply the
        # crossover operator to the parents. Otherwise, just take the
        # parents unchanged.
        if random.random() < self.crossover_probability:
            condition1, condition2 = parent1.condition.crossover_with(
                parent2.condition
            )
        else:
            condition1, condition2 = parent1.condition, parent2.condition

        # Apply the mutation operator to each child, randomly flipping
        # their mask bits with a small probability.
        condition1 = self._mutate(condition1, action_set.situation)
        condition2 = self._mutate(condition2, action_set.situation)

        # If the newly generated children are already present in the
        # population (or if they should be subsumed due to GA subsumption)
        # then simply increment the numerosities of the existing rules in
        # the population.
        new_children = []
        for condition in condition1, condition2:
            # If the parameters specify that GA subsumption should be
            # performed, look for an accurate parent that can subsume the
            # new child.
            if self.do_ga_subsumption:
                subsumed = False
                for parent in parent1, parent2:
                    should_subsume = (
                        (parent.experience >
                         self.subsumption_threshold) and
                        parent.error < self.error_threshold and
                        parent.condition(condition)
                    )
                    if should_subsume:
                        if parent in action_set.model:
                            parent.numerosity += 1
                            self.prune(action_set.model)
                        else:
                            # Sometimes the parent is removed from a
                            # previous subsumption
                            parent.numerosity = 1
                            action_set.model.add(parent)
                        subsumed = True
                        break
                if subsumed:
                    continue

            # Provided the child has not already been subsumed and it is
            # present in the population, just increment its numerosity.
            # Otherwise, if the child has neither been subsumed nor does it
            # already exist, remember it so we can add it to the classifier
            # set in just a moment.
            child = XCSClassifierRule(
                condition,
                action_set.action,
                self,
                action_set.model.time_stamp
            )
            if child in action_set.model:
                action_set.model.add(child)
            else:
                new_children.append(child)

        # If there were any children which weren't subsumed and weren't
        # already present in the classifier set, add them.
        if new_children:
            average_reward = .5 * (
                parent1.average_reward +
                parent2.average_reward
            )

            error = .5 * (parent1.error + parent2.error)

            # .1 * (average fitness of parents)
            fitness = .05 * (
                parent1.fitness +
                parent2.fitness
            )

            for child in new_children:
                child.average_reward = average_reward
                child.error = error
                child.fitness = fitness
                action_set.model.add(child)