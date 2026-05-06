def delete_applications(self, applications):
        """
        Requires: account ID, application ID (or name).
        Input should be a dictionary { 'app_id': 1234 , 'app': 'My Application'}
        Returns:  list of failed deletions (if any)
        Endpoint: api.newrelic.com
        Errors: None Explicit, failed deletions will be in XML
        Method: Post
        """
        endpoint = "https://api.newrelic.com"
        uri = "{endpoint}/api/v1/accounts/{account_id}/applications/delete.xml"\
              .format(endpoint=endpoint, account_id=self.account_id)
        payload = applications
        response = self._make_post_request(uri, payload)
        failed_deletions = {}

        for application in response.findall('.//application'):
            if not 'deleted' in application.findall('.//result')[0].text:
                failed_deletions['app_id'] = application.get('id')

        return failed_deletions