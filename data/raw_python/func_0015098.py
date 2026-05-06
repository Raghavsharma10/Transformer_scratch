def _get_call_params(self, method, **kwargs):
        """
        Returns the prepared call parameters. Mind, these will be keyword
        arguments to ``requests.post``.

        ``method`` the NVP method
        ``kwargs`` the actual call parameters
        """
        payload = {'METHOD': method,
                   'VERSION': self.config.API_VERSION}
        certificate = None

        if self.config.API_AUTHENTICATION_MODE == "3TOKEN":
            payload['USER'] = self.config.API_USERNAME
            payload['PWD'] = self.config.API_PASSWORD
            payload['SIGNATURE'] = self.config.API_SIGNATURE
        elif self.config.API_AUTHENTICATION_MODE == "CERTIFICATE":
            payload['USER'] = self.config.API_USERNAME
            payload['PWD'] = self.config.API_PASSWORD
            certificate = (self.config.API_CERTIFICATE_FILENAME,
                           self.config.API_KEY_FILENAME)
        elif self.config.API_AUTHENTICATION_MODE == "UNIPAY":
            payload['SUBJECT'] = self.config.UNIPAY_SUBJECT

        none_configs = [config for config, value in payload.items()
                        if value is None]
        if none_configs:
            raise PayPalConfigError(
                "Config(s) %s cannot be None. Please, check this "
                "interface's config." % none_configs)

        # all keys in the payload must be uppercase
        for key, value in kwargs.items():
            payload[key.upper()] = value

        return {'data': payload,
                'cert': certificate,
                'url': self.config.API_ENDPOINT,
                'timeout': self.config.HTTP_TIMEOUT,
                'verify': self.config.API_CA_CERTS}