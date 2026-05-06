def get_most_recent_update_time(self):
        """
        Indicated most recent update of the instance, assumption based on:
        - if currentWorkflow exists, its startedAt time is most recent update.
        - else max of workflowHistory startedAt is most recent update.
        """
        def parse_time(t):
            if t:
                return time.gmtime(t/1000)
            return None
        try:
            max_wf_started_at = max([i.get('startedAt') for i in self.workflowHistory])
            return parse_time(max_wf_started_at)
        except ValueError:
            return None