def show_domain_record(self, domain_id, record_id):
        """
        This method returns the specified domain record.

        Required parameters

            domain_id:
                Integer or Domain Name (e.g. domain.com), specifies the domain
                for which to retrieve a record.

            record_id:
                Integer, specifies the record_id to retrieve.
        """
        json = self.request('/domains/%s/records/%s' % (domain_id, record_id),
                            method='GET')
        status = json.get('status')
        if status == 'OK':
            domain_record_json = json.get('record')
            domain_record = Record.from_json(domain_record_json)
            return domain_record
        else:
            message = json.get('message')
            raise DOPException('[%s]: %s' % (status, message))