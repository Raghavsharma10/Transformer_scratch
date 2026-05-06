def get_kinds_data():
        """
        Returns a dict of all the data about the kinds, keyed to the kind
        value. e.g:
            {
                'gig': {
                    'name': 'Gig',
                    'slug': 'gigs',
                    'name_plural': 'Gigs',
                },
                # etc
            }
        """
        kinds = {k:{'name':v} for k,v in Event.KIND_CHOICES}
        for k,data in kinds.items():
            kinds[k]['slug'] = Event.KIND_SLUGS[k]
            kinds[k]['name_plural'] = Event.get_kind_name_plural(k)
        return kinds