def _sieve(self, multiple_records):
        """Return json object without multiple returns per resolved name.\
Names with multiple records are reduced by finding the name in the clade of\
interest, have the highest score, have the lowest taxonomic rank (if lowrank is
true) and/or are the first item returned."""
        # TODO: Break up, too complex
        GnrStore = self._store

        def writeAsJson(term, results):
            record = {'supplied_name_string': term}
            if len(results) > 0:
                record['results'] = results
            return record

        def boolResults(results, bool_li, rand=False):
            if rand:
                # choose first record (most likely best?)
                results = [results[0]]
            elif sum(bool_li) == 1:
                results = [results[bool_li.index(1)]]
            elif sum(bool_li) == 0:
                # return 'no_record'
                return []
            else:
                results = [result for i, result in enumerate(results) if
                           bool_li[i]]
            return results

        sieved = []
        ranks = ['species', 'genus', 'family', 'order', 'superorder', 'class',
                 'superclass', 'subphylum', 'phylum', 'kingdom',
                 'superkingdom']
        for term in multiple_records:
            results = GnrStore[term]
            while len(results) > 1:
                # choose result with best score
                scores = [result['score'] for result in results]
                bool_score = [1 if score == max(scores) else 0 for score in
                              scores]
                results = boolResults(results, bool_score)
                # choose result resolved to lowest taxonomic rank
                if self.lowrank:
                    res_ranks = [result['classification_path_ranks'].
                                 split('|') for result in results]
                    # calculate 'rank scores' for named and un-named ranks
                    nmd_rnks = []
                    unnmd_rnks = []
                    for rs in res_ranks:
                        nmd_rnks.append(min([j for j,e in enumerate(ranks) if
                                             e in rs]))
                        unnmd_rnk = [j for j,e in enumerate(rs) if
                                     e == ranks[nmd_rnks[-1]]][0]
                        unnmd_rnk -= len(rs)
                        unnmd_rnks.append(unnmd_rnk)
                    # calculate bool
                    unnmd_rnks = [e if nmd_rnks[j] == min(nmd_rnks) else 0 for
                                  j,e in enumerate(unnmd_rnks)]
                    bool_rank = [1 if e == min(unnmd_rnks) else 0 for e in
                                 unnmd_rnks]
                    results = boolResults(results, bool_rank)
                results = boolResults(results, [], rand=True)
            record = writeAsJson(term, results)
            sieved.append(record)
        return sieved