def run(self, scenario):
        """Run the algorithm, utilizing a classifier set to choose the
        most appropriate action for each situation produced by the
        scenario. Improve the situation/action mapping on each reward
        cycle to maximize reward. Return the classifier set that was
        created.

        Usage:
            scenario = MUXProblem()
            model = algorithm.run(scenario)

        Arguments:
            scenario: A Scenario instance.
        Return:
            A new classifier set, trained on the given scenario.
        """
        assert isinstance(scenario, scenarios.Scenario)
        model = self.new_model(scenario)
        model.run(scenario, learn=True)
        return model