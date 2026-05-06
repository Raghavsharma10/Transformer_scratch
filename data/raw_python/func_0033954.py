def create_domain(self, name, ip_address):
        """
        This method creates a new domain name with an A record for the specified [ip_address].

        Required parameters

            name:
                String, the name you want to give this SSH key.

            ip_address:
                String, ip address for the domain's initial a record.
        """
        params = {'name': name, 'ip_address': ip_address}
        json = self.request('/domains/new', method='GET', params=params)
        status = json.get('status')
        if status == 'OK':
            domain_json = json.get('domain')
            domain = Domain.from_json(domain_json)
            return domain
        else:
            message = json.get('message')
            raise DOPException('[%s]: %s' % (status, message))