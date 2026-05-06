def _linkToParent(self, feature, parentName):
        """
        Link a feature with its children
        """
        parentParts = self.byFeatureName.get(parentName)
        if parentParts is None:
            raise GFF3Exception(
                "Parent feature does not exist: {}".format(parentName),
                self.fileName)
        # parent maybe disjoint
        for parentPart in parentParts:
            feature.parents.add(parentPart)
            parentPart.children.add(feature)