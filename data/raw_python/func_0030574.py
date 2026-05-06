def is_indexed(self, identifier):
        """ Returns True if identifier is already indexed. Otherwise returns False. """
        with self.index.searcher() as searcher:
            result = searcher.search(Term('identifier', identifier['identifier']))
            return bool(result)