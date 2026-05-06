def _generateInitialModel(self, output_model_type):
        """Initialize using an MLHMM.

        """
        logger().info("Generating initial model for BHMM using MLHMM...")
        from bhmm.estimators.maximum_likelihood import MaximumLikelihoodEstimator
        mlhmm = MaximumLikelihoodEstimator(self.observations, self.nstates, reversible=self.reversible,
                                           output=output_model_type)
        model = mlhmm.fit()
        return model