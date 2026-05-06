def _get_instances(self, page_number=None):
        """
        Returns the service instances activated in this space.
        """
        instances = []
        uri = '/v2/spaces/%s/service_instances' % self.guid
        json_response = self.api.get(uri)
        instances += json_response['resources']
        while json_response['next_url'] is not None:
            json_response = self.api.get(json_response['next_url'])
            instances += json_response['resources']

        return instances