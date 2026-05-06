def count(self, query, _or=False):
        """Count word from FM-index
        Params:
            <str> | <Sequential> query
            <bool> _or
            <list <str> > ignores
        Return:
            <int> counts
        """
        if isinstance(query, str):
            return self.fm.count(query, MapIntInt({}))
        else:
            search_results = []
            for q in query:
                dids = MapIntInt({})
                self.fm.search(q, dids)
                search_results.append(dids.asdict())
            merged_dids = self._merge_search_result(search_results, _or)
            counts = 0
            for did in merged_dids:
                if _or:
                    counts += reduce(add, [int(x.pop(did, 0)) for x in search_results])
                else:
                    counts += min([int(x.pop(did, 0)) for x in search_results])
            return counts