def tokenize(self, message, max_length, mentions=None):
        """
        Tokenize a message into a list of messages of no more than max_length, including mentions
        in each message
        :param message: Message to be sent
        :param max_length: Maximum allowed length for each resulting message
        :param mentions: List of usernames to mention in each message
        :return:
        """
        mention_text = ''
        mention_length = 0
        if mentions:
            formatted_mentions = ['@{0}'.format(mention) for mention in mentions]
            mention_text = " ".join(formatted_mentions)
            message = '{0} {1}'.format(mention_text, message)
            mention_length = len(mention_text) + 1
        if len(message) <= max_length:
            return [message]

        tokens = message.split(' ')
        indices = []
        index = 1
        length = len(tokens[0])
        while index < len(tokens):
            # 1 for leading space, 4 for trailing " ..."
            if length + 1 + len(tokens[index]) + 4 > max_length:
                indices.append(index)
                # 4 for leading "... "
                length = 4 + mention_length + len(tokens[index])
            else:
                # 1 for leading space
                length += 1 + len(tokens[index])
            index += 1
        indices.append(index)

        messages = [" ".join(tokens[0:indices[0]])]
        for i in range(1, len(indices)):
            messages[i - 1] += ' ...'
            parts = []
            if mention_text:
                parts.append(mention_text)
            parts.append("...")
            parts.extend(tokens[indices[i - 1]:indices[i]])
            messages.append(" ".join(parts))

        return messages