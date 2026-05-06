def delay_response_series(self):
        """A tuple of two numpy arrays, which hold the time delays and the
        associated iuh values respectively."""
        delays = []
        responses = []
        sum_responses = 0.
        for t in itertools.count(self.dt_response/2., self.dt_response):
            delays.append(t)
            response = self(t)
            responses.append(response)
            sum_responses += self.dt_response*response
            if (sum_responses > .9) and (response < self.smallest_response):
                break
        return numpy.array(delays), numpy.array(responses)