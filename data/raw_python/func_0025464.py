def total_cost_function(self, item_a, item_b, time_a, time_b):
        """
        Calculate total cost function between two items.

        Args:
            item_a: STObject
            item_b: STObject
            time_a: Timestep in item_a at which cost function is evaluated
            time_b: Timestep in item_b at which cost function is evaluated

        Returns:
            The total weighted distance between item_a and item_b
        """
        distances = np.zeros(len(self.weights))
        for c, component in enumerate(self.cost_function_components):
            distances[c] = component(item_a, time_a, item_b, time_b, self.max_values[c])
        total_distance = np.sum(self.weights * distances)
        return total_distance