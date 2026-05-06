def put_record(self, data, partition_key=None):
        """Add data to the record queue in the proper format.

        Parameters
        ----------
        data : str
            Data to send.
        partition_key: str
            Hash that determines which shard a given data record belongs to.

        """
        # Byte encode the data
        data = encode_data(data)

        # Create a random partition key if not provided
        if not partition_key:
            partition_key = uuid.uuid4().hex

        # Build the record
        record = {
            'Data': data,
            'PartitionKey': partition_key
        }

        # Flush the queue if it reaches the batch size
        if self.queue.qsize() >= self.batch_size:
            logger.info("Queue Flush: batch size reached")
            self.pool.submit(self.flush_queue)

        # Append the record
        logger.debug('Putting record "{}"'.format(record['Data'][:100]))
        self.queue.put(record)