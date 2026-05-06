def send_records(self, records, attempt=0):
        """Send records to the Kinesis stream.

        Falied records are sent again with an exponential backoff decay.

        Parameters
        ----------
        records : array
            Array of formated records to send.
        attempt: int
            Number of times the records have been sent without success.
        """

        # If we already tried more times than we wanted, save to a file
        if attempt > self.max_retries:
            logger.warning('Writing {} records to file'.format(len(records)))
            with open('failed_records.dlq', 'ab') as f:
                for r in records:
                    f.write(r.get('Data'))
            return

        # Sleep before retrying
        if attempt:
            time.sleep(2 ** attempt * .1)

        response = self.kinesis_client.put_records(StreamName=self.stream_name,
                                                   Records=records)
        failed_record_count = response['FailedRecordCount']

        # Grab failed records
        if failed_record_count:
            logger.warning('Retrying failed records')
            failed_records = []
            for i, record in enumerate(response['Records']):
                if record.get('ErrorCode'):
                    failed_records.append(records[i])

            # Recursive call
            attempt += 1
            self.send_records(failed_records, attempt=attempt)