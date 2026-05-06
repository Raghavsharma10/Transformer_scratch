def search(self, id_key=None, **parameters):
        """ Searches TVDb for movie metadata

        TODO: Consider making parameters for episode ids
        """
        episode = parameters.get("episode")
        id_tvdb = parameters.get("id_tvdb") or id_key
        id_imdb = parameters.get("id_imdb")
        season = parameters.get("season")
        series = parameters.get("series")
        date = parameters.get("date")

        if id_tvdb:
            for result in self._search_id_tvdb(id_tvdb, season, episode):
                yield result
        elif id_imdb:
            for result in self._search_id_imdb(id_imdb, season, episode):
                yield result
        elif series and date:
            if not match(
                r"(19|20)\d{2}(-(?:0[1-9]|1[012])(-(?:[012][1-9]|3[01]))?)?",
                date,
            ):
                raise MapiProviderException("Date must be in YYYY-MM-DD format")
            for result in self._search_series_date(series, date):
                yield result
        elif series:
            for result in self._search_series(series, season, episode):
                yield result
        else:
            raise MapiNotFoundException