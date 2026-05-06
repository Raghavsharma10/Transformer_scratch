def send_with_media(
            self,
            *,
            text: str,
            files: List[str],
            captions: List[str]=[],
    ) -> List[OutputRecord]:
        """
        Upload media to mastodon,
        and send status and media,
        and captions if present.

        :param text: post text.
        :param files: list of files to upload with post.
        :param captions: list of captions to include as alt-text with files.
        :returns: list of output records,
            each corresponding to either a single post,
            or an error.
        """
        try:
            self.ldebug(f"Uploading files {files}.")
            if captions is None:
                captions = []

            if len(files) > len(captions):
                captions.extend([self.default_caption_message] * (len(files) - len(captions)))

            media_dicts = []
            for i, file in enumerate(files):
                caption = captions[i]
                media_dicts.append(self.api.media_post(file, description=caption))

            self.ldebug(f"Media ids {media_dicts}")

        except mastodon.MastodonError as e:
            return [self.handle_error(
                f"Bot {self.bot_name} encountered an error when uploading {files}:\n{e}\n", e
            )]

        try:
            status = self.api.status_post(status=text, media_ids=media_dicts)
            self.ldebug(f"Status object from toot: {status}.")
            return [TootRecord(record_data={
                "toot_id": status["id"],
                "text": text,
                "media_ids": media_dicts,
                "captions": captions
            })]

        except mastodon.MastodonError as e:
            return [self.handle_error((f"Bot {self.bot_name} encountered an error when "
                                      f"sending post {text} with media dicts {media_dicts}:"
                                      f"\n{e}\n"),
                                     e)]