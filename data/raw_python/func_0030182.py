def index_dataset(self, dataset, force=False):
        """ Adds given dataset to the index. """
        self.backend.dataset_index.index_one(dataset, force=force)