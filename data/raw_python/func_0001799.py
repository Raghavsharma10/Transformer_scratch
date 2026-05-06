def get_set(self, project, articleset, **filters):
        """List the articlesets in a project"""
        url = URL.articleset.format(**locals())
        return self.request(url, **filters)