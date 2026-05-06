def query_source(self, source):
        """
        Query by source
        """
        return self._get_repo_filter(Layer.objects).filter(url=source)