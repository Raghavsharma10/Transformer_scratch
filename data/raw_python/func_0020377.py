def poll(self, timeout_ms=0, max_records=None):
        """Fetch data from assigned topics / partitions.
        
        - ``max_records`` (int): maximum number of records to poll. Default: Inherit value from max_poll_records.
        - ``timeout_ms`` (int): Milliseconds spent waiting in poll if data is not available in the buffer.
          If 0, returns immediately with any records that are available currently in the buffer, else returns empty.
          Must not be negative. Default: `0`
        """

        messages = self.consumer.poll(timeout_ms=timeout_ms, max_records=max_records)

        result = []
        for _, msg in messages.items():
            for item in msg:
                result.append(item)
        return result