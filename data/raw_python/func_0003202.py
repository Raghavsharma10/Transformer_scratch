def feature_importances_table(self):
        """Feature importances table
        """
        from . import table

        return table.feature_importances(self.estimator,
                                         feature_names=self.feature_names)