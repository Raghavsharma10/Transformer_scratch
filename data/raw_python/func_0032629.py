def fetch_all_records(self):
        r"""
        Returns a generator that yields all of the DNS records for the domain

        :rtype: generator of `DomainRecord`\ s
        :raises DOAPIError: if the API endpoint replies with an error
        """
        api = self.doapi_manager
        return map(self._record, api.paginate(self.record_url, 'domain_records'))