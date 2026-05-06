def search(self, id_key=None, **parameters):
        """ Searches TMDb for movie metadata
        """
        id_tmdb = parameters.get("id_tmdb") or id_key
        id_imdb = parameters.get("id_imdb")
        title = parameters.get("title")
        year = parameters.get("year")

        if id_tmdb:
            yield self._search_id_tmdb(id_tmdb)
        elif id_imdb:
            yield self._search_id_imdb(id_imdb)
        elif title:
            for result in self._search_title(title, year):
                yield result
        else:
            raise MapiNotFoundException