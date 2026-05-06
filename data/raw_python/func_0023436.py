def clean(self):
        """Clean queue items from a previous session.

        In case a previous session crashed and there are still some running
        entries in the queue ('running', 'stopping', 'killing'), we clean those
        and enqueue them again.
        """
        for _, item in self.queue.items():
            if item['status'] in ['paused', 'running', 'stopping', 'killing']:
                item['status'] = 'queued'
                item['start'] = ''
                item['end'] = ''