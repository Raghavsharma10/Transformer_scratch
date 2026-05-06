def get_cert_types(self):
        """
        Collect the certificate types that are available to the customer.

        :return: A list of dictionaries of certificate types
        :rtype: list
        """
        result = self.client.service.getCustomerCertTypes(authData=self.auth)

        if result.statusCode == 0:
            return jsend.success({'cert_types': result.types})
        else:
            return self._create_error(result.statusCode)