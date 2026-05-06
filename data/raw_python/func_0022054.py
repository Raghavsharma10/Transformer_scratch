def query_ids(self, ids):
        """
        Query by list of identifiers
        """

        results = self._get_repo_filter(Layer.objects).filter(uuid__in=ids).all()

        if len(results) == 0:  # try services
            results = self._get_repo_filter(Service.objects).filter(uuid__in=ids).all()

        return results