def get_countries(self):
        """
        Returns a list of dicts, one per country that has at least one Venue
        in it.

        Each dict has 'code' and 'name' elements.
        The list is sorted by the country 'name's.
        """
        qs = Venue.objects.values('country') \
                                    .exclude(country='') \
                                    .distinct() \
                                    .order_by('country')

        countries = []

        for c in qs:
            countries.append({
                'code': c['country'],
                'name': Venue.get_country_name(c['country'])
            })

        return sorted(countries, key=lambda k: k['name'])