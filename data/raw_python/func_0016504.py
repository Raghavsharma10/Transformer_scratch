def _update_fitness(self, action_set):
        """Update the fitness values of the rules belonging to this action
        set."""
        # Compute the accuracy of each rule. Accuracy is inversely
        # proportional to error. Below a certain error threshold, accuracy
        # becomes constant. Accuracy values range over (0, 1].
        total_accuracy = 0
        accuracies = {}
        for rule in action_set:
            if rule.error < self.error_threshold:
                accuracy = 1
            else:
                accuracy = (
                    self.accuracy_coefficient *
                    (rule.error / self.error_threshold) **
                    -self.accuracy_power
                )
            accuracies[rule] = accuracy
            total_accuracy += accuracy * rule.numerosity

        # On rare occasions we have zero total accuracy. This avoids a div
        # by zero
        total_accuracy = total_accuracy or 1

        # Use the relative accuracies of the rules to update their fitness
        for rule in action_set:
            accuracy = accuracies[rule]
            rule.fitness += (
                self.learning_rate *
                (accuracy * rule.numerosity / total_accuracy -
                 rule.fitness)
            )