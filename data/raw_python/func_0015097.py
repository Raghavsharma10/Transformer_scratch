def _call(self, method, **kwargs):
        """
        Wrapper method for executing all API commands over HTTP. This method is
        further used to implement wrapper methods listed here:

        https://www.x.com/docs/DOC-1374

        ``method`` must be a supported NVP method listed at the above address.
        ``kwargs`` the actual call parameters
        """
        post_params = self._get_call_params(method, **kwargs)
        payload = post_params['data']
        api_endpoint = post_params['url']

        # This shows all of the key/val pairs we're sending to PayPal.
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug('PayPal NVP Query Key/Vals:\n%s' % pformat(payload))

        http_response = requests.post(**post_params)
        response = PayPalResponse(http_response.text, self.config)
        logger.debug('PayPal NVP API Endpoint: %s' % api_endpoint)

        if not response.success:
            raise PayPalAPIResponseError(response)

        return response