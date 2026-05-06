def _expand_place_ids(self, terms):
        """ Lookups all of the place identifiers to get gvids

        Args:
            terms (str or unicode): terms to lookup

        Returns:
            str or list: given terms if no identifiers found, otherwise list of identifiers.
        """

        place_vids = []
        first_type = None

        for result in self.backend.identifier_index.search(terms):

            if not first_type:
                first_type = result.type

            if result.type != first_type:
                # Ignore ones that aren't the same type as the best match
                continue

            place_vids.append(result.vid)

        if place_vids:
            # Add the 'all region' gvids for the higher level
            all_set = set(itertools.chain.from_iterable(iallval(GVid.parse(x)) for x in place_vids))
            place_vids += list(str(x) for x in all_set)
            return place_vids
        else:
            return terms