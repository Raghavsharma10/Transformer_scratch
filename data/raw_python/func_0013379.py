def insertRnaQuantificationSet(self, rnaQuantificationSet):
        """
        Inserts a the specified rnaQuantificationSet into this repository.
        """
        try:
            models.Rnaquantificationset.create(
                id=rnaQuantificationSet.getId(),
                datasetid=rnaQuantificationSet.getParentContainer().getId(),
                referencesetid=rnaQuantificationSet.getReferenceSet().getId(),
                name=rnaQuantificationSet.getLocalId(),
                dataurl=rnaQuantificationSet.getDataUrl(),
                attributes=json.dumps(rnaQuantificationSet.getAttributes()))
        except Exception:
            raise exceptions.DuplicateNameException(
                rnaQuantificationSet.getLocalId(),
                rnaQuantificationSet.getParentContainer().getLocalId())