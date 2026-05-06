def submit(self, cert_type_name, csr, revoke_password, term, subject_alt_names='',
               server_type='OTHER'):
        """
        Submit a certificate request to Comodo.

        :param string cert_type_name: The full cert type name (Example: 'PlatinumSSL Certificate') the supported
                                      certificate types for your account can be obtained with the
                                      get_cert_types() method.
        :param string csr: The Certificate Signing Request (CSR)
        :param string revoke_password: A password for certificate revocation
        :param int term: The length, in years, for the certificate to be issued
        :param string subject_alt_names: Subject Alternative Names separated by a ",".
        :param string server_type: The type of server for the TLS certificate e.g 'Apache/ModSSL' full list available in
                                   ComodoCA.server_type (Default: OTHER)
        :return: The certificate_id and the normal status messages for errors.
        :rtype: dict
        """
        cert_types = self.get_cert_types()

        # If collection of cert types fails we simply pass the error back.
        if cert_types['status'] == 'error':
            return cert_types

        # We do this because we need to pass the entire cert type definition back to Comodo
        # not just the name.
        for cert_type in cert_types['data']['cert_types']:
            if cert_type.name == cert_type_name:
                cert_type_def = cert_type

        result = self.client.service.enroll(authData=self.auth, orgId=self.org_id, secretKey=self.secret_key,
                                            csr=csr, phrase=revoke_password, subjAltNames=subject_alt_names,
                                            certType=cert_type_def, numberServers=1,
                                            serverType=ComodoCA.formats[server_type], term=term, comments='')

        # Anything greater than 0 is the certificate ID
        if result > 0:
            return jsend.success({'certificate_id': result})
        # Anything else is an error
        else:
            return self._create_error(result)