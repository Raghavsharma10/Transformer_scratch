def assign_to_topic_partition(self, topic_partition=None):
        """Assign a list of TopicPartitions to this consumer.
        
        - ``partitions`` (list of `TopicPartition`): Assignment for this instance.
        """

        if isinstance(topic_partition, TopicPartition):
            topic_partition = [topic_partition]
        if not self._is_assigned(topic_partition):
            self.consumer.assign(topic_partition)