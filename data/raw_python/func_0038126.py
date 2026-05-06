def predict(self):
        """
        Computes all possible weighted average predictions and their variances

        Example::

            >>> from caspo import core, predict

            >>> networks = core.LogicalNetworkList.from_csv('behaviors.csv')
            >>> setup = core.Setup.from_json('setup.json')

            >>> predictor = predict.Predictor(networks, setup)
            >>> df = predictor.predict()

            >>> df.to_csv('predictions.csv'), index=False)


        Returns
        --------
        `pandas.DataFrame`_
            DataFrame with the weighted average predictions and variance of all readouts for each possible clamping


        .. _pandas.DataFrame: http://pandas.pydata.org/pandas-docs/stable/dsintro.html#dataframe
        """
        self._logger.info("Computing all predictions and their variance for %s logical networks...", len(self.networks))

        return self.networks.predictions(self.setup.filter(self.networks))