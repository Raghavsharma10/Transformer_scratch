def domain_records(self, domain_id):
        """
        This method returns all of your current domain records.

        Required parameters

            domain_id:
                Integer or Domain Name (e.g. domain.com), specifies the domain
                for which to retrieve records.
        """
        json = self.request('/domains/%s/records' % domain_id, method='GET')
        status = json.get('status')
        if status == 'OK':
            records_json = json.get('records', [])
            records = [Record.from_json(record) for record in records_json]
            return records
        else:
            message = json.get('message')
            raise DOPException('[%s]: %s' % (status, message))