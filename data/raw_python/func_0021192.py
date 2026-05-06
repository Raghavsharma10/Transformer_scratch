def add_job(self, queue_name, job, timeout=200, replicate=None, delay=None,
                retry=None, ttl=None, maxlen=None, asynchronous=None):
        """
        Add a job to a queue.

        ADDJOB queue_name job <ms-timeout> [REPLICATE <count>] [DELAY <sec>]
            [RETRY <sec>] [TTL <sec>] [MAXLEN <count>] [ASYNC]

        :param queue_name: is the name of the queue, any string, basically.
        :param job: is a string representing the job.
        :param timeout: is the command timeout in milliseconds.
        :param replicate: count is the number of nodes the job should be
            replicated to.
        :param delay: sec is the number of seconds that should elapse
            before the job is queued by any server.
        :param retry: sec period after which, if no ACK is received, the
            job is put again into the queue for delivery. If RETRY is 0,
            the job has an at-most-once delivery semantics.
        :param ttl: sec is the max job life in seconds. After this time,
            the job is deleted even if it was not successfully delivered.
        :param maxlen: count specifies that if there are already count
            messages queued for the specified queue name, the message is
            refused and an error reported to the client.
        :param asynchronous: asks the server to let the command return ASAP and
            replicate the job to other nodes in the background. The job
            gets queued ASAP, while normally the job is put into the queue
            only when the client gets a positive reply. Changing the name of this
            argument as async is reserved keyword in python 3.7

        :returns: job_id
        """
        command = ['ADDJOB', queue_name, job, timeout]

        if replicate:
            command += ['REPLICATE', replicate]
        if delay:
            command += ['DELAY', delay]
        if retry is not None:
            command += ['RETRY', retry]
        if ttl:
            command += ['TTL', ttl]
        if maxlen:
            command += ['MAXLEN', maxlen]
        if asynchronous:
            command += ['ASYNC']

        # TODO(canardleteer): we need to handle "-PAUSE" messages more
        # appropriately, for now it's up to the person using the library
        # to handle a generic ResponseError on their own.
        logger.debug("sending job - %s", command)
        job_id = self.execute_command(*command)
        logger.debug("sent job - %s", command)
        logger.debug("job_id: %s " % job_id)
        return job_id