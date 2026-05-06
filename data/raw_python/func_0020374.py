def seek_to_end(self, topic_partition=None):
        """Seek to the most recent available offset for partitions.
        
        - ``topic_partition``: Optionally provide specific `TopicPartitions`,
          otherwise default to all assigned partitions.
        """

        if isinstance(topic_partition, TopicPartition):
            self.consumer.seek_to_end(topic_partition)
        else:
            raise TypeError("topic_partition must be of type TopicPartition, create it with Create TopicPartition keyword.")