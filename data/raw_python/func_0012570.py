def rq_link(self):
        """Link to Django-RQ status page for this job"""
        if self.rq_job:
            url = reverse('rq_job_detail',
                          kwargs={'job_id': self.rq_id, 'queue_index': queue_index_by_name(self.rq_origin)})
            return '<a href="{}">{}</a>'.format(url, self.rq_id)