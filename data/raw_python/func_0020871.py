def save_list(self, list_name, emails):
        """
        Upload a list. The list import job is queued and will happen shortly after the API request.
        http://docs.sailthru.com/api/list
        @param list: list name
        @param emails: List of email values or comma separated string
        """
        data = {'list': list_name,
                'emails': ','.join(emails) if isinstance(emails, list) else emails}
        return self.api_post('list', data)