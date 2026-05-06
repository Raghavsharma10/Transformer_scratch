def get_certificate_signing_request(self, **kwargs):
        """Construct a certificate signing request (CSR) for signing by a
        certificate authority (CA).

        :param \*\*kwargs: See the REST API Guide on your array for the
                           documentation on the request:
                           **GET cert/certificate_signing_request**
        :type \*\*kwargs: optional

        :returns: A dictionary mapping "certificate_signing_request" to the CSR.
        :rtype: ResponseDict

        .. note::

            Requires use of REST API 1.3 or later.

            In version 1.12, purecert was expanded to allow manipulation
            of multiple certificates, by name.  To preserve backwards compatibility,
            the default name, if none is specified, for this version is 'management'
            which acts on the certificate previously managed by this command.

        """
        if self._rest_version >= LooseVersion("1.12"):
            return self._request("GET",
                "cert/certificate_signing_request/{0}".format(
                    kwargs.pop('name', 'management')), kwargs)
        else:
            return self._request("GET", "cert/certificate_signing_request", kwargs)