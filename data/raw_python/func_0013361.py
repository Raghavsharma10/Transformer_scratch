def removePhenotypeAssociationSet(self, phenotypeAssociationSet):
        """
        Remove a phenotype association set from the repo
        """
        q = models.Phenotypeassociationset.delete().where(
            models.Phenotypeassociationset.id ==
            phenotypeAssociationSet.getId())
        q.execute()