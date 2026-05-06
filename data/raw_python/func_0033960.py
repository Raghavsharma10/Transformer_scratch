def destroy_domain_record(self, domain_id, record_id):
        """
        This method deletes the specified domain record.

        Required parameters

            domain_id:
                Integer or Domain Name (e.g. domain.com), specifies the domain
                for which to destroy a record.

            record_id:
                Integer, specifies the record_id to destroy.
        """
        json = self.request('/domains/%s/records/%s/destroy' % (domain_id, record_id),
                            method='GET')
        status = json.get('status')
        return status