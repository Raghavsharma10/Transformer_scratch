def search(self, query, _or=False, ignores=[]):
        """Search word from FM-index
        Params:
            <str> | <Sequential> query
            <bool> _or
            <list <str> > ignores
        Return:
            <list>SEARCH_RESULT(<int> document_id,
                                <list <int> > counts
                                <str> doc)
        """
        if isinstance(query, str):
            dids = MapIntInt({})
            self.fm.search(query, dids)
            dids = dids.asdict()
            result = []
            for did in sorted(dids.keys()):
                doc = self.fm.get_document(did)
                if not any(ignore in doc for ignore in ignores):
                    count = dids[did]
                    result.append(SEARCH_RESULT(int(did), [count], doc))
            return result

        search_results = []
        for q in query:
            dids = MapIntInt({})
            self.fm.search(q, dids)
            search_results.append(dids.asdict())
        merged_dids = self._merge_search_result(search_results, _or)
        result = []
        for did in merged_dids:
            doc = self.fm.get_document(did)
            if not any(ignore in doc for ignore in ignores):
                counts = map(lambda x: int(x.pop(did, 0)), search_results)
                result.append(SEARCH_RESULT(int(did), list(counts), doc))
        return result