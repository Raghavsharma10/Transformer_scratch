def __make_another_index(self, list_of_entries, url=False, hs_admin=False):
        '''
        Find an index not yet used in the handle record and not reserved for
            any (other) special type.

        :param: list_of_entries: List of all entries to find which indices are
            used already.
        :param url: If True, an index for an URL entry is returned (1, unless
            it is already in use).
        :param hs_admin: If True, an index for HS_ADMIN is returned (100 or one
            of the following).
        :return: An integer.
        '''

        start = 2

        # reserved indices:
        reserved_for_url = set([1])
        reserved_for_admin = set(range(100, 200))
        prohibited_indices = reserved_for_url | reserved_for_admin

        if url:
            prohibited_indices = prohibited_indices - reserved_for_url
            start = 1
        elif hs_admin:
            prohibited_indices = prohibited_indices - reserved_for_admin
            start = 100

        # existing indices
        existing_indices = set()
        if list_of_entries is not None:
            for entry in list_of_entries:
                existing_indices.add(int(entry['index']))

        # find new index:
        all_prohibited_indices = existing_indices | prohibited_indices
        searchmax = max(start, max(all_prohibited_indices)) + 2
        for index in xrange(start, searchmax):
            if index not in all_prohibited_indices:
                return index