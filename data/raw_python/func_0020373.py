def seek_to_beginning(self, topic_partition=None):
        """Seek to the oldest available offset for partitions.
        
        - ``topic_partition``: Optionally provide specific TopicPartitions,
          otherwise default to all assigned partitions.
        """

        if isinstance(topic_partition, TopicPartition):
            self.consumer.seek_to_beginning(topic_partition)
        else:
            raise TypeError("topic_partition must be of type TopicPartition, create it with Create TopicPartition keyword.")