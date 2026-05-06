def get_job(self, queues, timeout=None, count=None, nohang=False, withcounters=False):
        """
        Return some number of jobs from specified queues.

        GETJOB [NOHANG] [TIMEOUT <ms-timeout>] [COUNT <count>] [WITHCOUNTERS] FROM
            queue1 queue2 ... queueN

        :param queues: name of queues

        :returns: list of tuple(job_id, queue_name, job), tuple(job_id, queue_name, job, nacks, additional_deliveries) or empty list
        :rtype: list
        """
        assert queues

        command = ['GETJOB']
        if nohang:
            command += ['NOHANG']
        if timeout:
            command += ['TIMEOUT', timeout]
        if count:
            command += ['COUNT', count]
        if withcounters:
            command += ['WITHCOUNTERS']

        command += ['FROM'] + queues
        results = self.execute_command(*command)
        if not results:
            return []

        if withcounters:
            return [(job_id, queue_name, job, nacks, additional_deliveries) for
                    job_id, queue_name, job, _, nacks, _, additional_deliveries in results]
        else:
            return [(job_id, queue_name, job) for
                    job_id, queue_name, job in results]