def revoke(self, cert_id, reason=''):
        """
        Revoke a certificate.

        :param int cert_id: The certificate ID
        :param str reason: Reason for revocation (up to 256 characters), can be blank: ''
        :return: The result of the operation, 'Successful' on success
        :rtype: dict
        """
        result = self.client.service.revoke(authData=self.auth, id=cert_id, reason=reason)

        if result == 0:
            return jsend.success()
        else:
            return self._create_error(result)