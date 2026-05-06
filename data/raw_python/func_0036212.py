def create(self, mention, max_message_length):
        """
        Create a message
        :param mention: JSON object containing mention details from Twitter (or an empty dict {})
        :param max_message_length: Maximum allowable length for created message
        :return: A random message created using a Markov chain generator
        """
        message = []

        def message_len():
            return sum([len(w) + 1 for w in message])

        while message_len() < max_message_length:
            message.append(self.a_random_word(message[-1] if message else None))

        return ' '.join(message[:-1])