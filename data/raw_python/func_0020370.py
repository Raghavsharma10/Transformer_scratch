def subscribe_topic(self, topics=[], pattern=None):
        """Subscribe to a list of topics, or a topic regex pattern.
        
        - ``topics`` (list): List of topics for subscription.
        - ``pattern`` (str): Pattern to match available topics. You must provide either topics or pattern,
          but not both.
        """

        if not isinstance(topics, list):
            topics = [topics]
        self.consumer.subscribe(topics, pattern=pattern)