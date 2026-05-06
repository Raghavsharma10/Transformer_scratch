def regions(self):
        """
        This method will return all the available regions within the
        DigitalOcean cloud.
        """
        json = self.request('/regions', method='GET')
        status = json.get('status')
        if status == 'OK':
            regions_json = json.get('regions', [])
            regions = [Region.from_json(region) for region in regions_json]
            return regions
        else:
            message = json.get('message')
            raise DOPException('[%s]: %s' % (status, message))