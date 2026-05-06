def genotypesPhenotypesGenerator(self, request):
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
        featureSets = dataset.getFeatureSets()
        annotationList = phenotypeAssociationSet.getAssociations(
            request, featureSets)
        return self._protocolListGenerator(request, annotationList)