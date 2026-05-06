def get_pull_requests(self):
        "https://developer.github.com/v3/pulls/#list-pull-requests"
        g = self.github
        query = {'state': 'all'}
        if self.args.github_token:
            query['access_token'] = g['token']

        def f(pull):
            if self.args.ignore_closed:
                return (pull['state'] == 'opened' or
                        (pull['state'] == 'closed' and pull['merged_at']))
            else:
                return True
        pulls = filter(f,
                       self.get(g['url'] + "/repos/" + g['repo'] + "/pulls",
                                query, self.args.cache))
        return dict([(str(pull['number']), pull) for pull in pulls])