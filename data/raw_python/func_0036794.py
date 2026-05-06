def supported_providers(self):
        """
        Request a list of all available providers

        :returns: A list of all available providers (e.g. {'provider':
        'ec2_ap_northeast', 'title': 'EC2 AP NORTHEAST'})
        """
        req = self.request(self.uri + '/providers', api_version=2)
        providers = req.get().json()
        supported_providers = providers['supported_providers']
        return supported_providers