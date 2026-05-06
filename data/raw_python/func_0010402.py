def transmission_rate(self):
        """
        Returns the upstream, downstream values as a tuple in bytes per
        second. Use this for periodical calling.
        """
        sent = self.bytes_sent
        received = self.bytes_received
        traffic_call = time.time()
        time_delta = traffic_call - self.last_traffic_call
        upstream = int(1.0 * (sent - self.last_bytes_sent)/time_delta)
        downstream = int(1.0 * (received - self.last_bytes_received)/time_delta)
        self.last_bytes_sent = sent
        self.last_bytes_received = received
        self.last_traffic_call = traffic_call
        return upstream, downstream