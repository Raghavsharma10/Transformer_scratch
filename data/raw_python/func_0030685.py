def done(self, *args, **kwargs):
        """Mark the whole ProgressSection as done"""
        kwargs['state'] = 'done'
        pr_id = self.add(*args, log_action='done', **kwargs)

        self._session.query(Process).filter(Process.group == self._group).update({Process.state: 'done'})
        self.start.state = 'done'
        self._session.commit()

        return pr_id