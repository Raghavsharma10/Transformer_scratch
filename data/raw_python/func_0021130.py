def lookups(self, request, model_admin):
        """
        Returns a list of tuples like:

            [
                ('AU', 'Australia'),
                ('GB', 'UK'),
                ('US', 'USA'),
            ]

        One for each country that has at least one Venue.
        Sorted by the label names.
        """
        list_of_countries = []

        # We don't need the country_count but we need to annotate them in order
        # to group the results.
        qs = Venue.objects.exclude(country='') \
                            .values('country') \
                            .annotate(country_count=Count('country')) \
                            .order_by('country')
        for obj in qs:
            country = obj['country']
            list_of_countries.append(
                (country, Venue.COUNTRIES[country])
            )

        return sorted(list_of_countries, key=lambda c: c[1])