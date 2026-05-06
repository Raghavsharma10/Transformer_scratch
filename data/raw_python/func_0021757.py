def add_to_queue(self, series):
        """Add a series to the queue

        @param crunchyroll.models.Series series
        @return bool
        """
        result = self._android_api.add_to_queue(series_id=series.series_id)
        return result