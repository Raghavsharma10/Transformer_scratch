def _select_parent(action_set):
        """Select a rule from this action set, with probability
        proportionate to its fitness, to act as a parent for a new rule in
        the classifier set. Return the selected rule."""
        total_fitness = sum(rule.fitness for rule in action_set)
        selector = random.uniform(0, total_fitness)
        for rule in action_set:
            selector -= rule.fitness
            if selector <= 0:
                return rule
        # If for some reason a case slips through the above loop, perhaps
        # due to floating point error, we fall back on uniform selection.
        return random.choice(list(action_set))