def flush(self):
        """ Add accumulator to the moving average queue and reset it. For
        example, called by the StatsCollector once per second to calculate
        per-second average.
        """
        n = self.accumulator
        self.accumulator = 0

        stream = self.stream
        stream.append(n)
        self.sum += n

        streamlen = len(stream)

        if streamlen > self.period:
            self.sum -= stream.popleft()
            streamlen -= 1

        if streamlen == 0:
            self.last_average = 0
        else:
            self.last_average = self.sum / streamlen