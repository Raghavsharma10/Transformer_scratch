def is_indexed(self, dataset):
        """ Returns True if dataset is already indexed. Otherwise returns False. """
        with self.index.searcher() as searcher:
            result = searcher.search(Term('vid', dataset.vid))
            return bool(result)