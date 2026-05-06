def get_async_job(self, job_id):
        """Query an asynchronous SCI job by ID

        This is useful if the job was not created with send_sci_async().

        :param int job_id: The job ID to query
        :returns: The SCI response from GETting the job information
        """
        uri = "/ws/sci/{0}".format(job_id)
        # TODO: do parsing here?
        return self._conn.get(uri)