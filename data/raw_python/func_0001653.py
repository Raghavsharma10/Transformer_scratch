def status(self, job_id):
        """Gets the status of a previously-submitted job.
        """
        check_jobid(job_id)

        queue = self._get_queue()
        if queue is None:
            raise QueueDoesntExist

        ret, output = self._call('%s %s' % (
                                 shell_escape(queue / 'commands/status'),
                                 job_id),
                                 True)
        if ret == 0:
            directory, result = output.splitlines()
            result = result.decode('utf-8')
            return RemoteQueue.JOB_DONE, PosixPath(directory), result
        elif ret == 2:
            directory = output.splitlines()[0]
            return RemoteQueue.JOB_RUNNING, PosixPath(directory), None
        elif ret == 3:
            raise JobNotFound
        else:
            raise RemoteCommandFailure(command="commands/status",
                                       ret=ret)