def _linkFeature(self, feature):
        """
        Link a feature with its parents.
        """
        parentNames = feature.attributes.get("Parent")
        if parentNames is None:
            self.roots.add(feature)
        else:
            for parentName in parentNames:
                self._linkToParent(feature, parentName)