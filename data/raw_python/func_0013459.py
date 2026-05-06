def phenotypesGenerator(self, request):
        """
        Returns a generator over the (phenotypes, nextPageToken) pairs
        defined by the (JSON string) request
        """
        # TODO make paging work using SPARQL?
        compoundId = datamodel.PhenotypeAssociationSetCompoundId.parse(
            request.phenotype_association_set_id)
        dataset = self.getDataRepository().getDataset(compoundId.dataset_id)
        phenotypeAssociationSet = dataset.getPhenotypeAssociationSet(
            compoundId.phenotypeAssociationSetId)
        associations = phenotypeAssociationSet.getAssociations(request)
        phenotypes = [association.phenotype for association in associations]
        return self._protocolListGenerator(
            request, phenotypes)