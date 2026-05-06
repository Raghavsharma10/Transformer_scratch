def _process_params(self):
        """
        Adds default row size if it's not given in the query.
        Converts param values into unicode strings.

        Returns:
            Processed self._solr_params dict.
        """
        # transform sort dict into str
        self._sort_to_str()

        if 'rows' not in self._solr_params:
            self._solr_params['rows'] = self._cfg['row_size']

        for key, val in self._solr_params.items():
            if isinstance(val, str) and six.PY2:
                self._solr_params[key] = val.encode(encoding='UTF-8')
        return self._solr_params