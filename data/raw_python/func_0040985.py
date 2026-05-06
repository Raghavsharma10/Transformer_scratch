def create_job(self, emails):
        """
        Create a new bulk verification job for the list of emails.
        :param list emails: Email addresses to verify.
        :return: A Job object.
        """
        resp = self._call(endpoint='bulk', data={'input_location': '1', 'input': '\n'.join(emails)})
        return Job(resp['job_id'])