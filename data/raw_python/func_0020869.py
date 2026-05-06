def schedule_blast_from_blast(self, blast_id, schedule_time, options=None):
        """
        Schedule a mass mail blast from previous blast
        http://docs.sailthru.com/api/blast
        @param blast_id: blast_id to copy from
        @param schedule_time
        @param options: additional optional params
        """
        options = options or {}
        data = options.copy()
        data['copy_blast'] = blast_id
        data['schedule_time'] = schedule_time
        return self.api_post('blast', data)