def search_anime_series(self, query_string):
        """Search anime series list by series name, case-sensitive

        @param str query_string     string to search for, note that the search
                                        is very simplistic and only matches against
                                        the start of the series name, ex) search
                                        for "space" matches "Space Brothers" but
                                        wouldn't match "Brothers Space"
        @return list<crunchyroll.models.Series>
        """
        result = self._android_api.list_series(
            media_type=ANDROID.MEDIA_TYPE_ANIME,
            filter=ANDROID.FILTER_PREFIX + query_string)
        return result