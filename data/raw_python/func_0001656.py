def list(self):
        """Lists the jobs on the server.
        """
        queue = self._get_queue()
        if queue is None:
            raise QueueDoesntExist

        output = self.check_output('%s' %
                                   shell_escape(queue / 'commands/list'))

        job_id, info = None, None
        for line in output.splitlines():
            line = line.decode('utf-8')
            if line.startswith('    '):
                key, value = line[4:].split(': ', 1)
                info[key] = value
            else:
                if job_id is not None:
                    yield job_id, info
                job_id = line
                info = {}
        if job_id is not None:
            yield job_id, info