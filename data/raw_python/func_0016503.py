def prune(self, model):
        """Reduce the classifier set's population size, if necessary, by
        removing lower-quality *rules. Return a list containing any rules
        whose numerosities dropped to zero as a result of this call. (The
        list may be empty, if no rule's numerosity dropped to 0.) The
        model argument is a ClassifierSet instance which utilizes this
        algorithm.

        Usage:
            deleted_rules = model.algorithm.prune(model)

        Arguments:
            model: A ClassifierSet instance whose population may need to
                be reduced in size.
        Return:
            A possibly empty list of ClassifierRule instances which were
            removed entirely from the classifier set because their
            numerosities dropped to 0.
        """

        assert isinstance(model, ClassifierSet)
        assert model.algorithm is self

        # Determine the (virtual) population size.
        total_numerosity = sum(rule.numerosity for rule in model)

        # If the population size is already small enough, just return early
        if total_numerosity <= self.max_population_size:
            return []  # No rule's numerosity dropped to zero.

        # Determine the average fitness of the rules in the population.
        total_fitness = sum(rule.fitness for rule in model)
        average_fitness = total_fitness / total_numerosity

        # Determine the probability of deletion, as a function of both
        # accuracy and niche sparsity.
        total_votes = 0
        deletion_votes = {}
        for rule in model:
            vote = rule.action_set_size * rule.numerosity

            sufficient_experience = (
                rule.experience > self.deletion_threshold
            )
            low_fitness = (
                rule.fitness / rule.numerosity <
                self.fitness_threshold * average_fitness
            )
            if sufficient_experience and low_fitness:
                vote *= average_fitness / (rule.fitness /
                                           rule.numerosity)

            deletion_votes[rule] = vote
            total_votes += vote

        # Choose a rule to delete based on the probabilities just computed.
        selector = random.uniform(0, total_votes)
        for rule, vote in deletion_votes.items():
            selector -= vote
            if selector <= 0:
                assert rule in model
                if model.discard(rule):
                    return [rule]
                else:
                    return []

        assert False