def get_features2(self):
        """
        Return all features with its names.

        Returns
        -------
        names : list
            Feature names.
        values : list
            Feature values
        """

        feature_names = []
        feature_values = []

        # Get all the names of features.
        all_vars = vars(self)
        for name in all_vars.keys():
            # Omit input variables such as date, mag, err, etc.
            if not (name == 'date' or name == 'mag' or name == 'err'
                    or name == 'n_threads' or name == 'min_period'):
                # Filter some other unnecessary features.
                if not (name == 'f' or name == 'f_phase'
                        or name == 'period_log10FAP'
                        or name == 'weight' or name == 'weighted_sum'
                        or name == 'median' or name == 'mean' or name == 'std'):
                    feature_names.append(name)

        # Sort by the names.
        # Sorting should be done to keep maintaining the same order of features.
        feature_names.sort()

        # Get feature values.
        for name in feature_names:
            feature_values.append(all_vars[name])

        return feature_names, feature_values