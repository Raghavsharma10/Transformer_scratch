def insertPhenotypeAssociationSet(self, phenotypeAssociationSet):
        """
        Inserts the specified phenotype annotation set into this repository.
        """
        datasetId = phenotypeAssociationSet.getParentContainer().getId()
        attributes = json.dumps(phenotypeAssociationSet.getAttributes())
        try:
            models.Phenotypeassociationset.create(
                id=phenotypeAssociationSet.getId(),
                name=phenotypeAssociationSet.getLocalId(),
                datasetid=datasetId,
                dataurl=phenotypeAssociationSet._dataUrl,
                attributes=attributes)
        except Exception:
            raise exceptions.DuplicateNameException(
                phenotypeAssociationSet.getParentContainer().getId())