def list_media(self, series, sort=META.SORT_DESC, limit=META.MAX_MEDIA, offset=0):
        """List media for a given series or collection

        @param crunchyroll.models.Series series the series to search for
        @param str sort                         choose the ordering of the
                                                    results, only META.SORT_DESC
                                                    is known to work
        @param int limit                        limit size of results
        @param int offset                       start results from this index,
                                                    for pagination
        @return list<crunchyroll.models.Media>
        """
        params = {
            'sort': sort,
            'offset': offset,
            'limit': limit,
        }
        params.update(self._get_series_query_dict(series))
        result = self._android_api.list_media(**params)
        return result