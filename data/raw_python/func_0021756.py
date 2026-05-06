def list_queue(self, media_types=[META.TYPE_ANIME, META.TYPE_DRAMA]):
        """List the series in the queue, optionally filtering by type of media

        @param list<str> media_types    a list of media types to filter the queue
                                            with, should be of META.TYPE_*
        @return list<crunchyroll.models.Series>
        """
        result = self._android_api.queue(media_types='|'.join(media_types))
        return [queue_item['series'] for queue_item in result]