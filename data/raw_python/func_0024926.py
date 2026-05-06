def send(self, name=None, value=None, **kwargs):
        """
        Can accept a name/tag and value to be queued and then send anything in
        the queue to the time series service.  Optional parameters include
        setting quality, timestamp, or attributes.

        See spec for queue() for complete list of options.

        Example of sending a batch of values:

            queue('temp', 70.1)
            queue('humidity', 20.4)
            send()

        Example of sending one and flushing queue immediately

            send('temp', 70.3)
            send('temp', 70.4, quality=ts.GOOD, attributes={'unit': 'F'})


        """
        if name and value:
            self.queue(name, value, **kwargs)

        timestamp = int(round(time.time() * 1000))

        # The label "name" or "tag" is sometimes used ambiguously
        msg = {
            "messageId": timestamp,
            "body": self._queue
        }

        self._queue = []

        return self._send_to_timeseries(msg)