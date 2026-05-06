def new_model(self, scenario):
        """Create and return a new classifier set initialized for handling
        the given scenario.

        Usage:
            scenario = MUXProblem()
            model = algorithm.new_model(scenario)
            model.run(scenario, learn=True)

        Arguments:
            scenario: A Scenario instance.
        Return:
            A new, untrained classifier set, suited for the given scenario.
        """
        assert isinstance(scenario, scenarios.Scenario)
        return ClassifierSet(self, scenario.get_possible_actions())