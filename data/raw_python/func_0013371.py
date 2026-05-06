def insertVariantAnnotationSet(self, variantAnnotationSet):
        """
        Inserts a the specified variantAnnotationSet into this repository.
        """
        analysisJson = json.dumps(
            protocol.toJsonDict(variantAnnotationSet.getAnalysis()))
        try:
            models.Variantannotationset.create(
                id=variantAnnotationSet.getId(),
                variantsetid=variantAnnotationSet.getParentContainer().getId(),
                ontologyid=variantAnnotationSet.getOntology().getId(),
                name=variantAnnotationSet.getLocalId(),
                analysis=analysisJson,
                annotationtype=variantAnnotationSet.getAnnotationType(),
                created=variantAnnotationSet.getCreationTime(),
                updated=variantAnnotationSet.getUpdatedTime(),
                attributes=json.dumps(variantAnnotationSet.getAttributes()))
        except Exception as e:
            raise exceptions.RepoManagerException(e)