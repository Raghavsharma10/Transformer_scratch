def send(
            self,
            *,
            text: str,
    ) -> List[OutputRecord]:
        """
        Send mastodon message.

        :param text: text to send in post.
        :returns: list of output records,
            each corresponding to either a single post,
            or an error.
        """
        try:
            status = self.api.status_post(status=text)

            return [TootRecord(record_data={
                "toot_id": status["id"],
                "text": text
            })]

        except mastodon.MastodonError as e:
            return [self.handle_error((f"Bot {self.bot_name} encountered an error when "
                                      f"sending post {text} without media:\n{e}\n"),
                                     e)]