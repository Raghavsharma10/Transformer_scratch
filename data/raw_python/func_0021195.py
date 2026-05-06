def show(self, job_id, return_dict=False):
        """
        Describe the job.

        :param job_id:

        """
        rtn = self.execute_command('SHOW', job_id)

        if return_dict:
            grouped = self._grouper(rtn, 2)
            rtn = dict((a, b) for a, b in grouped)

        return rtn