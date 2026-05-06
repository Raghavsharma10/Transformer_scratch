def send(
            self,
            *,
            text: str,
    ) -> List[OutputRecord]:
        """
        Send birdsite message.

        :param text: text to send in post.
        :returns: list of output records,
            each corresponding to either a single post,
            or an error.
        """
        try:
            status = self.api.update_status(text)
            self.ldebug(f"Status object from tweet: {status}.")
            return [TweetRecord(record_data={"tweet_id": status._json["id"], "text": text})]

        except tweepy.TweepError as e:
            return [self.handle_error(
                message=(f"Bot {self.bot_name} encountered an error when "
                 f"sending post {text} without media:\n{e}\n"),
                error=e)]