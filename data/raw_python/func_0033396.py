def add_features(self, features):
        """Add features to this namespace.
        features: An iterable of features.  A feature may be either
            1) A VW label (not containing characters from escape_dict.keys(),
                unless 'escape' mode is on)
            2) A tuple (label, value) where value is any float
        """
        for feature in features:
            if isinstance(feature, basestring):
                label = feature
                value = None
            else:
                label, value = feature
            self.add_feature(label, value)