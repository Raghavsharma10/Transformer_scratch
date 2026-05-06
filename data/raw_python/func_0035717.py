def send_message(self, message, mention_id=None, mentions=[]):
        """
        Send the specified message to twitter, with appropriate mentions, tokenized as necessary
        :param message: Message to be sent
        :param mention_id: In-reply-to mention_id (to link messages to a previous message)
        :param mentions: List of usernames to mention in reply
        :return:
        """
        messages = self.tokenize(message, self.MESSAGE_LENGTH, mentions)
        code = 0
        for message in messages:
            if self.dry_run:
                mention_message = ''
                if mention_id:
                    mention_message = " to mention_id '{0}'".format(mention_id)
                logging.info("Not posting to Twitter because DRY_RUN is set. Would have posted "
                             "the following message{0}:\n{1}".format(mention_message, message))
            else:
                try:
                    self.twitter.statuses.update(status=message,
                                                 in_reply_to_status_id=mention_id)
                except TwitterHTTPError as e:
                    logging.error('Unable to post to twitter: {0}'.format(e))
                    code = e.response_data['errors'][0]['code']
        return code