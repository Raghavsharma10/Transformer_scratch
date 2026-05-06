def kill(self, job_id):
        """Kills a job on the server.
        """
        check_jobid(job_id)

        queue = self._get_queue()
        if queue is None:
            raise QueueDoesntExist

        ret, output = self._call('%s %s' % (
                                 shell_escape(queue / 'commands/kill'),
                                 job_id),
                                 False)
        if ret == 3:
            raise JobNotFound
        elif ret != 0:
            raise RemoteCommandFailure(command='commands/kill',
                                       ret=ret)