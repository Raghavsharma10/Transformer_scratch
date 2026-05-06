def runGetExpressionLevel(self, id_):
        """
        Runs a getExpressionLevel request for the specified ID.
        """
        compoundId = datamodel.ExpressionLevelCompoundId.parse(id_)
        dataset = self.getDataRepository().getDataset(compoundId.dataset_id)
        rnaQuantificationSet = dataset.getRnaQuantificationSet(
            compoundId.rna_quantification_set_id)
        rnaQuantification = rnaQuantificationSet.getRnaQuantification(
            compoundId.rna_quantification_id)
        expressionLevel = rnaQuantification.getExpressionLevel(compoundId)
        return self.runGetRequest(expressionLevel)