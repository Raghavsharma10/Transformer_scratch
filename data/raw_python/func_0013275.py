def getVariantAnnotationSet(self, id_):
        """
        Returns the AnnotationSet in this dataset with the specified 'id'
        """
        if id_ not in self._variantAnnotationSetIdMap:
            raise exceptions.AnnotationSetNotFoundException(id_)
        return self._variantAnnotationSetIdMap[id_]