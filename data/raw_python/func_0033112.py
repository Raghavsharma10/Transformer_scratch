def get_merge_requests(self):
        "http://doc.gitlab.com/ce/api/merge_requests.html"
        g = self.gitlab
        merges = self.get(g['url'] + "/projects/" +
                          g['repo'] + "/merge_requests",
                          {'private_token': g['token'],
                           'state': 'all'}, cache=False)
        return dict([(str(merge['id']), merge) for merge in merges])