def get_dupe_prob(self, url):
        """ A probability of given url being a duplicate of some content
        that has already been seem.
        """
        path, query = _parse_url(url)
        dupestats = []
        extend_ds = lambda x: dupestats.extend(filter(None, (
            ds_dict.get(key) for ds_dict, key in x)))
        if self.urls_by_path.get(path):
            extend_ds([(self.path_dupstats, path)])
        # If param is in the query
        for param, value in query.items():
            qwp_key = _q_key(_without_key(query, param))
            # Have we seen the query with param changed or removed?
            has_changed = self.urls_by_path_qwp.get((path, param, qwp_key))
            has_removed = self.urls_by_path_q.get((path, qwp_key))
            if has_changed or has_removed:
                extend_ds(self._param_dupstats(path, param, qwp_key))
            if has_removed:
                extend_ds(self._param_value_dupstats(path, param, value))
        # If param is not in the query, but we've crawled a page when it is
        q_key = _q_key(query)
        for param in (self.params_by_path.get(path, set()) - set(query)):
            if self.urls_by_path_qwp.get((path, param, q_key)):
                extend_ds(self._param_dupstats(path, param, q_key))
                # FIXME - this could be a long list of param values,
                # it's better to somehow store only high-probability values?
                for value in self.param_values.get((path, param), set()):
                    extend_ds(self._param_value_dupstats(path, param, value))
        return max(ds.get_prob() for ds in dupestats) if dupestats else 0.