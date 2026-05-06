def linkChildFeaturesToParents(self):
        """
        finish loading the set, constructing the tree
        """
        # features maybe disjoint
        for featureParts in self.byFeatureName.itervalues():
            for feature in featureParts:
                self._linkFeature(feature)