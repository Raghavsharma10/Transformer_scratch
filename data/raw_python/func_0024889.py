def create(self, parameters={}, **kwargs):
        """
        Create an instance of the US Weather Forecast Service with
        typical starting settings.
        """
        # Add parameter during create for UAA issuer
        uri = self.uaa.service.settings.data['uri'] + '/oauth/token'
        parameters["trustedIssuerIds"] = [uri]
        super(PredixService, self).create(parameters=parameters, **kwargs)