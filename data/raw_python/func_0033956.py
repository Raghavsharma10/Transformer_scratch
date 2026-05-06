def destroy_domain(self, domain_id):
        """
        This method deletes the specified domain.

        Required parameters

            domain_id:
                Integer or Domain Name (e.g. domain.com), specifies the domain
                to destroy.
        """
        json = self.request('/domains/%s/destroy' % domain_id, method='GET')
        status = json.get('status')
        return status