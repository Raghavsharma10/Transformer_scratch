def expressionLevelsGenerator(self, request):
        """
        Returns a generator over the (expressionLevel, nextPageToken) pairs
        defined by the specified request.

        Currently only supports searching over a specified rnaQuantification
        """
        rnaQuantificationId = request.rna_quantification_id
        compoundId = datamodel.RnaQuantificationCompoundId.parse(
            request.rna_quantification_id)
        dataset = self.getDataRepository().getDataset(compoundId.dataset_id)
        rnaQuantSet = dataset.getRnaQuantificationSet(
            compoundId.rna_quantification_set_id)
        rnaQuant = rnaQuantSet.getRnaQuantification(rnaQuantificationId)
        rnaQuantificationId = rnaQuant.getLocalId()
        iterator = paging.ExpressionLevelsIterator(
            request, rnaQuant)
        return iterator