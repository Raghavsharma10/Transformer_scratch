def search(self, search_phrase, limit=None):
        """Search for datasets, and expand to database records"""
        from ambry.identity import ObjectNumber
        from ambry.orm.exc import NotFoundError
        from ambry.library.search_backends.base import SearchTermParser

        results = []

        stp = SearchTermParser()

        # Because of the split between searching for partitions and bundles, some terms don't behave right.
        # The source term should be a limit on everything, but it isn't part of the partition doc,
        # so we check for it here.
        parsed_terms = stp.parse(search_phrase)

        for r in self.search_datasets(search_phrase, limit):
            vid = r.vid or ObjectNumber.parse(next(iter(r.partitions))).as_dataset

            r.vid = vid

            try:
                r.bundle = self.library.bundle(r.vid)

                if 'source' not in parsed_terms or parsed_terms['source'] in r.bundle.dataset.source:
                    results.append(r)
            except NotFoundError:
                pass

        return sorted(results, key=lambda r : r.score, reverse=True)