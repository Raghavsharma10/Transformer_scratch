def rq_job(self):
        """The last RQ Job this ran on"""
        if not self.rq_id or not self.rq_origin:
            return
        try:
            return RQJob.fetch(self.rq_id, connection=get_connection(self.rq_origin))
        except NoSuchJobError:
            return