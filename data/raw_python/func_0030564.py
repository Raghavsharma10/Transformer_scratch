def all(self):
        """ Returns list with all indexed datasets. """
        datasets = []
        for dataset in self.index.searcher().documents():
            res = DatasetSearchResult()
            res.vid = dataset['vid']
            res.b_score = 1
            datasets.append(res)
        return datasets