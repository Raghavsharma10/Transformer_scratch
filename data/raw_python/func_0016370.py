def put_records(self, records, partition_key=None):
        """Add a list of data records to the record queue in the proper format.
        Convinience method that calls self.put_record for each element.

        Parameters
        ----------
        records : list
            Lists of records to send.
        partition_key: str
            Hash that determines which shard a given data record belongs to.

        """
        for record in records:
            self.put_record(record, partition_key)