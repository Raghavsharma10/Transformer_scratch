def insertDataset(self, dataset):
        """
        Inserts the specified dataset into this repository.
        """
        try:
            models.Dataset.create(
                id=dataset.getId(),
                name=dataset.getLocalId(),
                description=dataset.getDescription(),
                attributes=json.dumps(dataset.getAttributes()))
        except Exception:
            raise exceptions.DuplicateNameException(
                dataset.getLocalId())