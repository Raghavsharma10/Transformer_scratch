def all(self):
        """ Returns list with all indexed datasets. """
        datasets = []

        query = text("""
            SELECT vid
            FROM dataset_index;""")

        for result in self.backend.library.database.connection.execute(query):
            res = DatasetSearchResult()
            res.vid = result[0]
            res.b_score = 1
            datasets.append(res)
        return datasets