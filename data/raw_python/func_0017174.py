def add_features(self, **features):
        """ Add or update several features. """
        for fname, fvalue in six.iteritems(features):
            setattr(self, fname, fvalue)
            self.features.add(fname)