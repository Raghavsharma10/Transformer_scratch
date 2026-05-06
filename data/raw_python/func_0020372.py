def seek(self, offset, topic_partition=None):
        """Manually specify the fetch offset for a TopicPartition.
        
        - ``offset``: Message offset in partition
        - ``topic_partition`` (`TopicPartition`): Partition for seek operation
        """

        if isinstance(topic_partition, TopicPartition):
            self.consumer.seek(topic_partition, offset=offset)
        else:
            raise TypeError("topic_partition must be of type TopicPartition, create it with Create TopicPartition keyword.")