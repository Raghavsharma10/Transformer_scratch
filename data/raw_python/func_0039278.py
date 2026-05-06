def fetch_all(self, org):
        """
        Fetches all local objects
        :param org: the org
        :return: the queryset
        """
        qs = self.model.objects.filter(org=org)
        if self.local_backend_attr is not None:
            qs = qs.filter(**{self.local_backend_attr: self.backend})
        return qs