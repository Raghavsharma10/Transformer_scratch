def _handle_caption_upload(
            self,
            *,
            media_ids: List[str],
            captions: Optional[List[str]],
    ) -> None:
        """
        Handle uploading all captions.

        :param media_ids: media ids of uploads to attach captions to.
        :param captions: captions to be attached to those media ids.
        :returns: None.
        """
        if captions is None:
            captions = []

        if len(media_ids) > len(captions):
            captions.extend([self.default_caption_message] * (len(media_ids) - len(captions)))

        for i, media_id in enumerate(media_ids):
            caption = captions[i]
            self._upload_caption(media_id=media_id, caption=caption)