def list_sets(self, project, **filters):
        """List the articlesets in a project"""
        url = URL.articlesets.format(**locals())
        return self.get_pages(url, **filters)