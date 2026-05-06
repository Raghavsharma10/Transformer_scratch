def perform_batch_reply(
            self,
            *,
            callback: Callable[..., str],
            lookback_limit: int,
            target_handle: str,
    ) -> List[OutputRecord]:
        """
        Performs batch reply on target account.
        Looks up the recent messages of the target user,
        applies the callback,
        and replies with
        what the callback generates.

        :param callback: a callback taking a message id,
            message contents,
            and optional extra keys,
            and returning a message string.
        :param target: the id of the target account.
        :param lookback_limit: a lookback limit of how many messages to consider.
        :returns: list of output records,
            each corresponding to either a single post,
            or an error.
        """
        self.log.info(f"Attempting to batch reply to birdsite user {target_handle}")

        if "@" in target_handle:
            base_target_handle = target_handle[1:]
        else:
            base_target_handle = target_handle

        records: List[OutputRecord] = []
        statuses = self.api.user_timeline(screen_name=base_target_handle, count=lookback_limit)
        self.log.debug(f"Retrieved {len(statuses)} statuses.")
        for i, status in enumerate(statuses):
            self.log.debug(f"Processing status {i} of {len(statuses)}")
            status_id = status.id

            # find possible replies we've made.
            # the 10 * lookback_limit is a guess,
            # might not be enough and I'm not sure we can guarantee it is.
            our_statuses = self.api.user_timeline(since_id=status_id,
                                                  count=lookback_limit * 10)
            in_reply_to_ids = list(map(lambda x: x.in_reply_to_status_id, our_statuses))

            if status_id not in in_reply_to_ids:
                # the twitter API and tweepy will attempt to give us the truncated text of the
                # message if we don't do this roundabout thing.
                encoded_status_text = self.api.get_status(status_id,
                                                  tweet_mode="extended")._json["full_text"]

                status_text = html.unescape(encoded_status_text)
                message = callback(message_id=status_id, message=status_text, extra_keys={})

                full_message = f"@{base_target_handle} {message}"
                self.log.info(f"Trying to reply with {message} to status {status_id} "
                              f"from {target_handle}.")
                try:
                    new_status = self.api.update_status(status=full_message,
                                                        in_reply_to_status_id=status_id)

                    records.append(TweetRecord(record_data={
                        "tweet_id": new_status.id,
                        "in_reply_to": f"@{base_target_handle}",
                        "in_reply_to_id": status_id,
                        "text": full_message,
                    }))

                except tweepy.TweepError as e:
                    records.append(self.handle_error(
                        message=(f"Bot {self.bot_name} encountered an error when "
                         f"trying to reply to {status_id} with {message}:\n{e}\n"),
                        error=e))
            else:
                self.log.info(f"Not replying to status {status_id} from {target_handle} "
                              f"- we already replied.")

        return records