def cancel(self, job_ids):
        ''' Cancels the jobs specified by a list of job ids

        Args:
        job_ids : [<job_id> ...]

        Returns :
        [True/False...] : If the cancel operation fails the entire list will be False.
        '''

        job_id_list = ' '.join(job_ids)
        retcode, stdout, stderr = self.channel.execute_wait("qdel {0}".format(job_id_list), 3)
        rets = None
        if retcode == 0:
            for jid in job_ids:
                self.resources[jid]['status'] = translate_table['E']  # Setting state to exiting
            rets = [True for i in job_ids]
        else:
            rets = [False for i in job_ids]

        return rets