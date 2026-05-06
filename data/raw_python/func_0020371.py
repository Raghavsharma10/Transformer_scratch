def get_position(self, topic_partition=None):
        """Return offset of the next record that will be fetched.
        
        - ``topic_partition`` (TopicPartition): Partition to check
        """

        if isinstance(topic_partition, TopicPartition):
            return self.consumer.position(topic_partition)
        else:
            raise TypeError("topic_partition must be of type TopicPartition, create it with Create TopicPartition keyword.")