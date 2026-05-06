def populateFromRow(self, annotationSetRecord):
        """
        Populates this VariantAnnotationSet from the specified DB row.
        """
        self._annotationType = annotationSetRecord.annotationtype
        self._analysis = protocol.fromJson(
            annotationSetRecord.analysis, protocol.Analysis)
        self._creationTime = annotationSetRecord.created
        self._updatedTime = annotationSetRecord.updated
        self.setAttributesJson(annotationSetRecord.attributes)