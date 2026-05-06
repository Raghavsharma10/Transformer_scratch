def _merge_search_result(self, search_results, _or=False):
        """Merge of filter search results
        Params:
            <str> | <Sequential> query
            <bool> _or
        Return:
            <list> computed_dids
        """
        all_docids = reduce(add, [list(x.keys()) for x in search_results])
        if _or:
            return sorted(set(all_docids), key=all_docids.index)
        return [docid for docid in set(all_docids) if all_docids.count(docid) > 1]