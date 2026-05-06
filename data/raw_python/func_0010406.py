def _body_builder(self, kwargs):
        """
        Helper method to construct the appropriate SOAP-body to call a
        FritzBox-Service.
        """
        p = {
            'action_name': self.name,
            'service_type': self.service_type,
            'arguments': '',
            }
        if kwargs:
            arguments = [
                self.argument_template % {'name': k, 'value': v}
                for k, v in kwargs.items()
            ]
            p['arguments'] = ''.join(arguments)
        body = self.body_template.strip() % p
        return body