def search(self, search_phrase, limit=None):
        """ Finds identifier by search phrase. """
        self._parsed_query = search_phrase
        schema = self._get_generic_schema()
        parser = QueryParser('name', schema=schema)
        query = parser.parse(search_phrase)

        class PosSizeWeighting(scoring.WeightingModel):

            def scorer(self, searcher, fieldname, text, qf=1):
                return self.PosSizeScorer(searcher, fieldname, text, qf=qf)

            class PosSizeScorer(scoring.BaseScorer):
                def __init__(self, searcher, fieldname, text, qf=1):
                    self.searcher = searcher
                    self.fieldname = fieldname
                    self.text = text
                    self.qf = qf
                    self.bmf25 = scoring.BM25F()

                def max_quality(self):
                    return 40

                def score(self, matcher):
                    poses = matcher.value_as('positions')
                    return (2.0 / (poses[0] + 1) + 1.0 / (len(self.text) / 4 + 1) +
                            self.bmf25.scorer(searcher, self.fieldname, self.text).score(matcher))

        with self.index.searcher(weighting=PosSizeWeighting()) as searcher:
            results = searcher.search(query, limit=limit)
            for hit in results:
                vid = hit['identifier']
                yield IdentifierSearchResult(
                    score=hit.score, vid=vid,
                    type=hit.get('type', False),
                    name=hit.get('name', ''))