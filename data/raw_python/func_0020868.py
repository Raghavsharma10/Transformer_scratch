def schedule_blast_from_template(self, template, list_name, schedule_time, options=None):
        """
        Schedule a mass mail blast from template
        http://docs.sailthru.com/api/blast
        @param template: template to copy from
        @param list_name: list to send to
        @param schedule_time
        @param options: additional optional params
        """
        options = options or {}
        data = options.copy()
        data['copy_template'] = template
        data['list'] = list_name
        data['schedule_time'] = schedule_time
        return self.api_post('blast', data)