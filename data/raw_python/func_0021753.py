def get_media_stream(self, media_item, format, quality):
        """Get the stream data for a given media item

        @param crunchyroll.models.Media media_item
        @param int format
        @param int quality
        @return crunchyroll.models.MediaStream
        """
        result = self._ajax_api.VideoPlayer_GetStandardConfig(
            media_id=media_item.media_id,
            video_format=format,
            video_quality=quality)
        return MediaStream(result)