def search_media(self, series, query_string):
        """Search for media from a series starting with query_string, case-sensitive

        @param crunchyroll.models.Series series     the series to search in
        @param str query_string                     the search query, same restrictions
                                                        as `search_anime_series`
        @return list<crunchyroll.models.Media>
        """
        params = {
            'sort': ANDROID.FILTER_PREFIX + query_string,
        }
        params.update(self._get_series_query_dict(series))
        result = self._android_api.list_media(**params)
        return result