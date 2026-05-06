def get_submissions(self, url):
        """
        Connects to Reddit and gets a JSON representation of submissions.
        This JSON data is then processed and returned.

        url: A url that requests for submissions should be sent to.
        """
        response = self.client.get(url, params={'limit': self.options['limit']})
        submissions = [x['data'] for x in response.json()['data']['children']]
        return submissions