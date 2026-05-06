def get_number_of_messages_in_topicpartition(self, topic_partition=None):
        """Return number of messages in TopicPartition.
        
        - ``topic_partition`` (list of TopicPartition)
        """

        if isinstance(topic_partition, TopicPartition):
            topic_partition = [topic_partition]

        number_of_messages = 0
        assignment = self.consumer.assignment()

        self.consumer.unsubscribe()
        for Partition in topic_partition:
            if not isinstance(Partition, TopicPartition):
                raise TypeError("topic_partition must be of type TopicPartition, create it with Create TopicPartition keyword.")

            self.assign_to_topic_partition(Partition)

            self.consumer.seek_to_end(Partition)
            end = self.consumer.position(Partition)
            self.consumer.seek_to_beginning(Partition)
            start = self.consumer.position(Partition)
            number_of_messages += end-start

        self.consumer.unsubscribe()
        self.consumer.assign(assignment)
        return number_of_messages