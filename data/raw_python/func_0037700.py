def evergreen(self, included_channel_ids=None, excluded_channel_ids=None, **kwargs):
        """
        Search containing any evergreen piece of Content.

        :included_channel_ids list: Contains ids for channel ids relevant to the query.
        :excluded_channel_ids list: Contains ids for channel ids excluded from the query.
        """
        eqs = self.search(**kwargs)
        eqs = eqs.filter(Evergreen())
        if included_channel_ids:
            eqs = eqs.filter(VideohubChannel(included_ids=included_channel_ids))
        if excluded_channel_ids:
            eqs = eqs.filter(VideohubChannel(excluded_ids=excluded_channel_ids))
        return eqs