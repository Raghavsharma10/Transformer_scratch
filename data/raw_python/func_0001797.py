def aggregate(self, **filters):
        """Conduct an aggregate query"""
        url = URL.aggregate.format(**locals())
        return self.get_pages(url, **filters)