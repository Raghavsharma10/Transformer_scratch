def _fill_in_cainfo(self):
        """Fill in the path of the PEM file containing the CA certificate.

        The priority is: 1. user provided path, 2. path to the cacert.pem
        bundle provided by certifi (if installed), 3. let pycurl use the
        system path where libcurl's cacert bundle is assumed to be stored,
        as established at libcurl build time.
        """
        if self.cainfo:
            cainfo = self.cainfo
        else:
            try:
                cainfo = certifi.where()
            except AttributeError:
                cainfo = None
        if cainfo:
            self._pycurl.setopt(pycurl.CAINFO, cainfo)