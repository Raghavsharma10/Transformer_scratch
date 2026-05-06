def remove_from_queue(self, series):
        """Remove a series from the queue

        @param crunchyroll.models.Series series
        @return bool
        """
        result = self._android_api.remove_from_queue(series_id=series.series_id)
        return result