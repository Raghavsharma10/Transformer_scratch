def fetch_local(self, org, identity):
        """
        Fetches the local model instance with the given identity, returning none if it doesn't exist
        :param org: the org
        :param identity: the unique identity
        :return: the instance or none
        """
        qs = self.fetch_all(org=org).filter(**{self.local_id_attr: identity})

        if self.select_related:
            qs = qs.select_related(*self.select_related)
        if self.prefetch_related:
            qs = qs.prefetch_related(*self.prefetch_related)

        return qs.first()