def create_domain_record(self, domain_id, record_type, data, name=None,
                             priority=None, port=None, weight=None):
        """
        This method creates a new domain name with an A record for the specified
        [ip_address].

        Required parameters

            domain_id:
                Integer or Domain Name (e.g. domain.com), specifies the domain
                for which to create a record.

            record_type:
                String, the type of record you would like to create.
                'A', 'CNAME', 'NS', 'TXT', 'MX' or 'SRV'

            data:
                String, this is the value of the record

        Optional parameters
            name:
                String, required for 'A', 'CNAME', 'TXT' and 'SRV' records

            priority:
                Integer, required for 'SRV' and 'MX' records

            port:
                Integer, required for 'SRV' records

            weight:
                Integer, required for 'SRV' records
        """
        params = dict(record_type=record_type, data=data)

        if name:
            params.update({'name': name})
        if priority:
            params.update({'priority': priority})
        if port:
            params.update({'port': port})
        if weight:
            params.update({'weight': weight})

        json = self.request('/domains/%s/records/new' % domain_id, method='GET', params=params)
        status = json.get('status')
        if status == 'OK':
            domain_record_json = json.get('domain_record')
            domain_record = Record.from_json(domain_record_json)
            return domain_record
        else:
            message = json.get('message')
            raise DOPException('[%s]: %s' % (status, message))