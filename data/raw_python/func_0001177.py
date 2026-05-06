def prefix_topic(self, topics):
        """
        Adds the topic_prefix to topic(s) supplied
        """
        if not self.topic_prefix or not topics:
            return topics

        if not isinstance(topics, str) and isinstance(topics, collections.Iterable):
            return [self.topic_prefix + topic for topic in topics]

        return self.topic_prefix + topics