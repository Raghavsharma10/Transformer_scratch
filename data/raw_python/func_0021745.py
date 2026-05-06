def list_anime_series(self, sort=META.SORT_ALPHA, limit=META.MAX_SERIES, offset=0):
        """Get a list of anime series

        @param str sort     pick how results should be sorted, should be one
                                of META.SORT_*
        @param int limit    limit number of series to return, there doesn't
                                seem to be an upper bound
        @param int offset   list series starting from this offset, for pagination
        @return list<crunchyroll.models.Series>
        """
        result = self._android_api.list_series(
            media_type=ANDROID.MEDIA_TYPE_ANIME,
            filter=sort,
            limit=limit,
            offset=offset)
        return result