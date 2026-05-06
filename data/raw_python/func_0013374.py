def insertFeatureSet(self, featureSet):
        """
        Inserts a the specified featureSet into this repository.
        """
        # TODO add support for info and sourceUri fields.
        try:
            models.Featureset.create(
                id=featureSet.getId(),
                datasetid=featureSet.getParentContainer().getId(),
                referencesetid=featureSet.getReferenceSet().getId(),
                ontologyid=featureSet.getOntology().getId(),
                name=featureSet.getLocalId(),
                dataurl=featureSet.getDataUrl(),
                attributes=json.dumps(featureSet.getAttributes()))
        except Exception as e:
            raise exceptions.RepoManagerException(e)