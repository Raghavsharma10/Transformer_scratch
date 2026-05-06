def feature_importances(self):
        """Feature importances plot
        """
        return plot.feature_importances(self.estimator,
                                        feature_names=self.feature_names,
                                        ax=_gen_ax())