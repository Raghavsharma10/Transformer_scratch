def removeDataset(self, dataset):
        """
        Removes the specified dataset from this repository. This performs
        a cascading removal of all items within this dataset.
        """
        for datasetRecord in models.Dataset.select().where(
                        models.Dataset.id == dataset.getId()):
            datasetRecord.delete_instance(recursive=True)