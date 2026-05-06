def runGetDataset(self, id_):
        """
        Runs a getDataset request for the specified ID.
        """
        dataset = self.getDataRepository().getDataset(id_)
        return self.runGetRequest(dataset)