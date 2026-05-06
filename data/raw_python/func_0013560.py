def removeVariantSet(self):
        """
        Removes a variantSet from the repo.
        """
        self._openRepo()
        dataset = self._repo.getDatasetByName(self._args.datasetName)
        variantSet = dataset.getVariantSetByName(self._args.variantSetName)

        def func():
            self._updateRepo(self._repo.removeVariantSet, variantSet)
        self._confirmDelete("VariantSet", variantSet.getLocalId(), func)