def _compute_prediction(self):
        """Compute the combined prediction and prediction weight for this
        action set. The combined prediction is the weighted average of the
        individual predictions of the classifiers. The combined prediction
        weight is the sum of the individual prediction weights of the
        classifiers.

        Usage:
            Do not call this method directly. Use the prediction and/or
            prediction_weight properties instead.

        Arguments: None
        Return: None
        """
        total_weight = 0
        total_prediction = 0
        for rule in self._rules.values():
            total_weight += rule.prediction_weight
            total_prediction += (rule.prediction *
                                 rule.prediction_weight)
        self._prediction = total_prediction / (total_weight or 1)
        self._prediction_weight = total_weight