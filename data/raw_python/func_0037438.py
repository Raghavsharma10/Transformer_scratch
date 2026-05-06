def update_model(self, url, text):
        """ Update prediction model with a page by given url and text content.
        Return a list of item duplicates (for testing purposes).
        """
        min_hash = get_min_hash(text, self.too_common_shingles, self.num_perm)
        item_url = canonicalize_url(url)
        item_path, item_query = _parse_url(item_url)
        all_duplicates = [
            (url, self.seen_urls[url]) for url in self.lsh.query(min_hash)]
        duplicates = [(url, m.query) for url, m in all_duplicates
                      if m.path == item_path]
        # Hypothesis (1) - just paths
        n_path_nodup = self._nodup_filter(min_hash, (
            self.urls_by_path.get(item_path, set())
            .difference(url for url, _ in duplicates)))
        self.path_dupstats[item_path].update(len(duplicates), n_path_nodup)
        # Other hypotheses, if param is in the query
        for param, value in item_query.items():
            self._update_with_param(
                duplicates, min_hash, item_path, item_query, param, [value])
        # Other hypotheses, if param is not in the query
        for param in (
                self.params_by_path.get(item_path, set()) - set(item_query)):
            self._update_with_param(
                duplicates, min_hash, item_path, item_query, param,
                self.param_values.get((item_path, param), set()))
        # Update indexes
        for param, value in item_query.items():
            self.urls_by_path_q[item_path, _q_key(item_query)].add(item_url)
            item_qwp_key = _q_key(_without_key(item_query, param))
            self.urls_by_path_qwp[item_path, param, item_qwp_key].add(item_url)
            self.params_by_path[item_path].add(param)
            self.param_values[item_path, param].add(value)
        if not item_query:
            self.urls_by_path_q[item_path, ()].add(item_url)
        self.urls_by_path[item_path].add(item_url)
        if item_url in self.lsh:
            self.lsh.remove(item_url)
        self.lsh.insert(item_url, min_hash)
        self.seen_urls[item_url] = URLMeta(item_path, item_query, min_hash)
        if len(self.seen_urls) % 100 == 0:
            self.log_dupstats()
        return all_duplicates