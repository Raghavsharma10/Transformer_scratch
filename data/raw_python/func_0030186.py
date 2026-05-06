def search_datasets(self, search_phrase, limit=None):
        """ Search for datasets. """
        return self.backend.dataset_index.search(search_phrase, limit=limit)