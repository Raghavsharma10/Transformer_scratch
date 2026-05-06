def insertContinuousSet(self, continuousSet):
        """
        Inserts a the specified continuousSet into this repository.
        """
        # TODO add support for info and sourceUri fields.
        try:
            models.ContinuousSet.create(
                id=continuousSet.getId(),
                datasetid=continuousSet.getParentContainer().getId(),
                referencesetid=continuousSet.getReferenceSet().getId(),
                name=continuousSet.getLocalId(),
                dataurl=continuousSet.getDataUrl(),
                attributes=json.dumps(continuousSet.getAttributes()))
        except Exception as e:
            raise exceptions.RepoManagerException(e)