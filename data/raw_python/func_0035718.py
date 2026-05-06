def get_reply_to_names(self, mention):
        """
        Get a sorted list of unique usernames mentioned in the message, excluding the bot's own name
        :param mention: JSON mention object from twitter
        :return: list of usernames
        """
        mention_list = [user['screen_name'] for user in mention['entities']['user_mentions']]
        mention_list.append(mention['user']['screen_name'])
        reply_to_names = set(mention_list)
        # Do not include bot's own name
        reply_to_names.discard(self.screen_name)
        return sorted(list(reply_to_names))