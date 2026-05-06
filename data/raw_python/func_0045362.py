def _sort_to_str(self):
        """
        Before exec query, this method transforms sort dict string

        from

            {"name": "asc", "timestamp":"desc"}

        to

            "name asc, timestamp desc"
        """

        params_list = []
        timestamp = ""

        for k, v in self._solr_params['sort'].items():
            if k != "timestamp":
                params_list.append(" ".join([k, v]))
            else:
                timestamp = v

        params_list.append(" ".join(['timestamp', timestamp]))

        self._solr_params['sort'] = ", ".join(params_list)