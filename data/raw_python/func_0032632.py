def update_record(self, **attrs):
        # The `_record` is to avoid conflicts with MutableMapping.update.
        """
        Update the record, modifying any number of its attributes (except
        ``id``).  ``update_record`` takes the same keyword arguments as
        :meth:`Domain.create_record`; pass in only those attributes that you
        want to update.

        :return: an updated `DomainRecord` object
        :rtype: DomainRecord
        :raises DOAPIError: if the API endpoint replies with an error
        """
        return self.domain._record(self.doapi_manager.request(self.url,
                                                              method='PUT',
                                                              data=attrs)\
                                                             ["domain_record"])