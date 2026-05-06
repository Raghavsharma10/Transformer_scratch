def create(self, command, **args):
        """
        Create a job given a command
        :param command: Nutch command, one of nutch.LegalJobs
        :param args: Additional arguments to pass to the job
        :return: The created Job
        """

        command = command.upper()
        if command not in LegalJobs:
            warn('Nutch command must be one of: %s' % ', '.join(LegalJobs))
        else:
            echo2('Starting %s job with args %s' % (command, str(args)))
        parameters = self.parameters.copy()
        parameters['type'] = command
        parameters['crawlId'] = self.crawlId
        parameters['confId'] = self.confId
        parameters['args'].update(args)

        job_info = self.server.call('post', "/job/create", parameters, JsonAcceptHeader)

        job = Job(job_info['id'], self.server)
        return job