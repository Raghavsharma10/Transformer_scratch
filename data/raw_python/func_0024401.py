def get_substructure_mapping(self, other, limit=1):
        """
        get self to other substructure mapping

        :param limit: number of matches. if 0 return iterator for all possible; if 1 return dict or None;
            if > 1 return list of dicts
        """
        i = self._matcher(other).subgraph_isomorphisms_iter()
        if limit == 1:
            m = next(i, None)
            if m:
                return {v: k for k, v in m.items()}
            return
        elif limit == 0:
            return ({v: k for k, v in m.items()} for m in i)
        return [{v: k for k, v in m.items()} for m in islice(i, limit)]