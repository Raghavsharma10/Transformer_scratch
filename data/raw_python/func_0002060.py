def _to_url(self):
        """ Serialises this query into a request-able URL including parameters """
        url = self._target_url

        params = collections.defaultdict(list, copy.deepcopy(self._filters))
        if self._order_by is not None:
            params['sort'] = self._order_by
        for k, vl in self._extra.items():
            params[k] += vl

        if params:
            url += "?" + urllib.parse.urlencode(params, doseq=True)

        return url