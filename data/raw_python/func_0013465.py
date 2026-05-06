def rnaQuantificationsGenerator(self, request):
        """
        Returns a generator over the (rnaQuantification, nextPageToken) pairs
        defined by the specified request.
        """
        if len(request.rna_quantification_set_id) < 1:
            raise exceptions.BadRequestException(
                "Rna Quantification Set Id must be specified")
        else:
            compoundId = datamodel.RnaQuantificationSetCompoundId.parse(
                request.rna_quantification_set_id)
            dataset = self.getDataRepository().getDataset(
                compoundId.dataset_id)
            rnaQuantSet = dataset.getRnaQuantificationSet(
                compoundId.rna_quantification_set_id)
        results = []
        for obj in rnaQuantSet.getRnaQuantifications():
            include = True
            if request.biosample_id:
                if request.biosample_id != obj.getBiosampleId():
                    include = False
            if include:
                results.append(obj)
        return self._objectListGenerator(request, results)