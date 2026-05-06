def _updateEmissionProbabilities(self):
        """Sample a new set of emission probabilites from the conditional distribution P(E | S, O)

        """
        observations_by_state = [self.model.collect_observations_in_state(self.observations, state)
                                 for state in range(self.model.nstates)]
        self.model.output_model.sample(observations_by_state)