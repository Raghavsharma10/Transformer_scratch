def get_stream_formats(self, media_item):
        """Get the available media formats for a given media item

        @param crunchyroll.models.Media
        @return dict
        """
        scraper = ScraperApi(self._ajax_api._connector)
        formats = scraper.get_media_formats(media_item.media_id)
        return formats