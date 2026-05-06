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
        self.log.info(f"Attempting to batch reply to mastodon user {target_handle}")

        # target handle should be able to be provided either as @user or @user@domain
        # note that this produces an empty first chunk
        handle_chunks = target_handle.split("@")
        target_base_handle = handle_chunks[1]

        records: List[OutputRecord] = []
        our_id = self.api.account_verify_credentials()["id"]

        # be careful here - we're using a search to do this,
        # and if we're not careful we'll pull up people just mentioning the target.
        possible_accounts = self.api.account_search(target_handle, following=True)
        their_id = None
        for account in possible_accounts:
            if account["username"] == target_base_handle:
                their_id = account["id"]
                break

        if their_id is None:
            return [self.handle_error(f"Could not find target handle {target_handle}!", None)]

        statuses = self.api.account_statuses(their_id, limit=lookback_limit)
        for status in statuses:

            status_id = status.id

            # find possible replies we've made.
            our_statuses = self.api.account_statuses(our_id, since_id=status_id)
            in_reply_to_ids = list(map(lambda x: x.in_reply_to_id, our_statuses))
            if status_id not in in_reply_to_ids:

                encoded_status_text = re.sub(self.html_re, "", status.content)
                status_text = html.unescape(encoded_status_text)

                message = callback(message_id=status_id, message=status_text, extra_keys={})
                self.log.info(f"Replying {message} to status {status_id} from {target_handle}.")
                try:
                    new_status = self.api.status_post(status=message, in_reply_to_id=status_id)

                    records.append(TootRecord(record_data={
                        "toot_id": new_status.id,
                        "in_reply_to": target_handle,
                        "in_reply_to_id": status_id,
                        "text": message,
                    }))

                except mastodon.MastodonError as e:
                    records.append(
                        self.handle_error((f"Bot {self.bot_name} encountered an error when "
                                           f"sending post {message} during a batch reply "
                                           f":\n{e}\n"),
                                          e))
            else:
                self.log.info(f"Not replying to status {status_id} from {target_handle} "
                              f"- we already replied.")

        return records