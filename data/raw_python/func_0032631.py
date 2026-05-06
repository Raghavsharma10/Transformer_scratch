def fetch(self):
        """
        Fetch & return a new `DomainRecord` object representing the domain
        record's current state

        :rtype: DomainRecord
        :raises DOAPIError: if the API endpoint replies with an error (e.g., if
            the domain record no longer exists)
        """
        return self.domain._record(self.doapi_manager.request(self.url)\
                                                             ["domain_record"])