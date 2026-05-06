def get_number_of_messages_in_topics(self, topics):
        """Retrun number of messages in topics.
        
        - ``topics`` (list): list of topics.
        """

        if not isinstance(topics, list):
            topics = [topics]

        number_of_messages = 0
        for t in topics:
            part = self.get_kafka_partitions_for_topic(topic=t)
            Partitions = map(lambda p: TopicPartition(topic=t, partition=p), part)
            number_of_messages += self.get_number_of_messages_in_topicpartition(Partitions)

        return number_of_messages