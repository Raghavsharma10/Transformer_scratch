def create_record(self, type, name, data, priority=None, port=None,
                      weight=None, **kwargs):
        # pylint: disable=redefined-builtin
        """
        Add a new DNS record to the domain

        :param str type: the type of DNS record to add (``"A"``, ``"CNAME"``,
            etc.)
        :param str name: the name (hostname, alias, etc.) of the new record
        :param str data: the value of the new record
        :param int priority: the priority of the new record (SRV and MX records
            only)
        :param int port: the port that the service is accessible on (SRV
            records only)
        :param int weight: the weight of records with the same priority (SRV
            records only)
        :param kwargs: additional fields to include in the API request
        :return: the new domain record
        :rtype: DomainRecord
        :raises DOAPIError: if the API endpoint replies with an error
        """
        api = self.doapi_manager
        data = {
            "type": type,
            "name": name,
            "data": data,
            "priority": priority,
            "port": port,
            "weight": weight,
        }
        data.update(kwargs)
        return self._record(api.request(self.record_url, method='POST',
                                        data=data)["domain_record"])