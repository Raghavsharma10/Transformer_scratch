def list_articles(self, project, articleset, page=1, **filters):
        """List the articles in a set"""
        url = URL.article.format(**locals())
        return self.get_pages(url, page=page, **filters)