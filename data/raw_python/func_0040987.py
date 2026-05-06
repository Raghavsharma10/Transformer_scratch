def results(self, job_id):
        """
        Yield the result of a completed bulk verification job.
        :param int job_id: ID of a job to retrieve the results for.
        :yields: The next VerifiedEmail objects.
        """
        resp = self._call(endpoint='download', data={'job_id': job_id})
        Row = namedtuple('Row', ['email', 'result_text_code'])
        for line in resp:
            row = Row(*line.decode('utf-8').split(','))
            yield VerifiedEmail.from_text_code(row.email, row.result_text_code)