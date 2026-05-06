def collect(self, cert_id, format_type):
        """
        Poll for certificate availability after submission.

        :param int cert_id: The certificate ID
        :param str format_type: The format type to use (example: 'X509 PEM Certificate only')
        :return: The certificate_id or the certificate depending on whether the certificate is ready (check status code)
        :rtype: dict
        """

        result = self.client.service.collect(authData=self.auth, id=cert_id,
                                             formatType=ComodoCA.format_type[format_type])

        # The certificate is ready for collection
        if result.statusCode == 2:
            return jsend.success({'certificate': result.SSL.certificate, 'certificate_status': 'issued',
                                  'certificate_id': cert_id})
        # The certificate is not ready for collection yet
        elif result.statusCode == 0:
            return jsend.fail({'certificate_id': cert_id, 'certificate': '', 'certificate_status': 'pending'})
        # Some error occurred
        else:
            return self._create_error(result.statusCode)