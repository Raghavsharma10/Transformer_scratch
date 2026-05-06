def addVariantAnnotationSet(self, variantAnnotationSet):
        """
        Adds the specified variantAnnotationSet to this dataset.
        """
        id_ = variantAnnotationSet.getId()
        self._variantAnnotationSetIdMap[id_] = variantAnnotationSet
        self._variantAnnotationSetIds.append(id_)