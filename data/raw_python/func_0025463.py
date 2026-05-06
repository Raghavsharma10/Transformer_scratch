def cost_matrix(self, set_a, set_b, time_a, time_b):
        """
        Calculates the costs (distances) between the items in set a and set b at the specified times.

        Args:
            set_a: List of STObjects
            set_b: List of STObjects
            time_a: time at which objects in set_a are evaluated
            time_b: time at whcih object in set_b are evaluated

        Returns:
            A numpy array with shape [len(set_a), len(set_b)] containing the cost matrix between the items in set a
            and the items in set b.
        """
        costs = np.zeros((len(set_a), len(set_b)))
        for a, item_a in enumerate(set_a):
            for b, item_b in enumerate(set_b):
                costs[a, b] = self.total_cost_function(item_a, item_b, time_a, time_b)
        return costs