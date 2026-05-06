def get_features_all(self):
        """
        Return all features with its names.

        Regardless of being used for train and prediction. Sorted by the names.

        Returns
        -------
        all_features : OrderedDict
            Features dictionary.
        """

        features = {}

        # Get all the names of features.
        all_vars = vars(self)
        for name in all_vars.keys():
            if name in feature_names_list_all:
                features[name] = all_vars[name]

        # Sort by the keys (i.e. feature names).
        features = OrderedDict(sorted(features.items(), key=lambda t: t[0]))

        return features