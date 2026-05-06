def domains(self):
        """
        This method returns all of your current domains.
        """
        json = self.request('/domains', method='GET')
        status = json.get('status')
        if status == 'OK':
            domains_json = json.get('domains', [])
            domains = [Domain.from_json(domain) for domain in domains_json]
            return domains
        else:
            message = json.get('message')
            raise DOPException('[%s]: %s' % (status, message))