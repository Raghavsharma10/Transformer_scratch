def search_manga_series(self, query_string):
        """Search the manga series list by name, case-insensitive

        @param str query_string

        @return list<crunchyroll.models.Series>
        """

        result = self._manga_api.list_series()
        return [series for series in result \
            if series['locale']['enUS']['name'].lower().startswith(
                query_string.lower())]