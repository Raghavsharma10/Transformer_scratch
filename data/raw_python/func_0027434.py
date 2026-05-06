def retrieve_info(self):
        """Query the Github API to retrieve the needed infos."""

        path = urlparse(self.url).path
        path = path.split('/')[1:]

        sanity_filter = re.compile('[\da-z-_]+', re.IGNORECASE)
        self.product = sanity_filter.match(path[0]).group(0)
        self.component = sanity_filter.match(path[1]).group(0)
        self.issue_id = int(path[3])

        github_url = '%s/%s/%s/issues/%s' % (_URL_BASE,
                                             self.product,
                                             self.component,
                                             self.issue_id)

        result = requests.get(github_url)
        self.status_code = result.status_code

        if result.status_code == 200:
            result = result.json()
            self.title = result['title']
            self.reporter = result['user']['login']
            if result['assignee'] is not None:
                self.assignee = result['assignee']['login']
            self.status = result['state']
            self.created_at = result['created_at']
            self.updated_at = result['updated_at']
            self.closed_at = result['closed_at']
        elif result.status_code == 404:
            self.title = 'private issue'