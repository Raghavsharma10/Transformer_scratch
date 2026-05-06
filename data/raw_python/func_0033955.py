def show_domain(self, domain_id):
        """
        This method returns the specified domain.

        Required parameters

            domain_id:
                Integer or Domain Name (e.g. domain.com), specifies the domain
                to display.
        """
        json = self.request('/domains/%s' % domain_id, method='GET')
        status = json.get('status')
        if status == 'OK':
            domain_json = json.get('domain')
            domain = Domain.from_json(domain_json)
            return domain
        else:
            message = json.get('message')
            raise DOPException('[%s]: %s' % (status, message))