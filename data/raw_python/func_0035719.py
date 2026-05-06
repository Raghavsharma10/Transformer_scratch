def reply_to_mentions(self):
        """
        For every mention since since_id, create a message with the provider and use it to
        reply to the mention
        :return: Number of mentions processed
        """
        since_id = self.since_id.get()

        kwargs = {'count': 200}
        if since_id:
            kwargs['since_id'] = since_id

        mentions_list = []
        try:
            mentions_list = self.twitter.statuses.mentions_timeline(**kwargs)
        except TwitterHTTPError as e:
            logging.error('Unable to retrieve mentions from twitter: {0}'.format(e))

        logging.info("Retrieved {0} mentions".format(len(mentions_list)))

        mentions_processed = 0
        # We want to process least recent to most recent, so that since_id is set properly
        for mention in reversed(mentions_list):
            mention_id = mention['id']
            reply_to_names = self.get_reply_to_names(mention)

            error_code = self.DUPLICATE_CODE
            tries = 0
            message = ''
            while error_code == self.DUPLICATE_CODE:
                if tries > 10:
                    logging.error('Unable to post duplicate message to {0}: {1}'.format(
                                  reply_to_names, message))
                    break
                elif tries == 10:
                    # Tried 10 times to post a message, but all were duplicates
                    message = 'No unique messages found.'
                else:
                    message = self.messages.create(mention, self.MESSAGE_LENGTH)
                error_code = self.send_message(message, mention_id, reply_to_names)
                tries += 1

            mentions_processed += 1
            self.since_id.set('{0}'.format(mention_id))

        return mentions_processed