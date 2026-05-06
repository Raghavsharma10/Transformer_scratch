def send_with_media(
            self,
            *,
            text: str,
            files: List[str],
            captions: List[str]=[]
    ) -> List[OutputRecord]:
        """
        Upload media to birdsite,
        and send status and media,
        and captions if present.

        :param text: tweet text.
        :param files: list of files to upload with post.
        :param captions: list of captions to include as alt-text with files.
        :returns: list of output records,
            each corresponding to either a single post,
            or an error.
        """

        # upload media
        media_ids = None
        try:
            self.ldebug(f"Uploading files {files}.")
            media_ids = [self.api.media_upload(file).media_id_string for file in files]
        except tweepy.TweepError as e:
            return [self.handle_error(
                message=f"Bot {self.bot_name} encountered an error when uploading {files}:\n{e}\n",
                error=e)]

        # apply captions, if present
        self._handle_caption_upload(media_ids=media_ids, captions=captions)

        # send status
        try:
            status = self.api.update_status(status=text, media_ids=media_ids)
            self.ldebug(f"Status object from tweet: {status}.")
            return [TweetRecord(record_data={
                "tweet_id": status._json["id"],
                "text": text,
                "media_ids": media_ids,
                "captions": captions,
                "files": files
            })]

        except tweepy.TweepError as e:
            return [self.handle_error(
                message=(f"Bot {self.bot_name} encountered an error when "
                 f"sending post {text} with media ids {media_ids}:\n{e}\n"),
                error=e)]